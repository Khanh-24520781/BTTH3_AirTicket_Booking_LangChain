"""BTVN#3: dữ liệu, công cụ mock và bốn lớp Harness dùng chung.
"""

from __future__ import annotations

from threading import RLock

import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import date as Date, datetime, time
from typing import Optional

from langchain_core.tools import tool


@dataclass(frozen=True)
class FlightConstraints:
    """Ràng buộc bất biến: model đề xuất hành động, không sửa mục tiêu."""

    origin: str = "SGN"
    destination: str = "DAD"
    date: str = "2026-10-07"
    depart_before: str = "12:00"
    max_price: int = 2_000_000
    passenger_name: str = "Nguyen Van A"

    def __post_init__(self):
        Date.fromisoformat(self.date)
        time.fromisoformat(self.depart_before)
        if not self.origin or not self.destination or not self.passenger_name.strip():
            raise ValueError("Chặng bay và hành khách không được để trống.")
        if type(self.max_price) is not int or self.max_price < 0:
            raise ValueError("Ngân sách phải là số nguyên không âm.")

    def to_prompt(self) -> str:
        return (f"Đặt 1 vé {self.origin} -> {self.destination} ngày {self.date}, "
                f"cất cánh trước {self.depart_before}, giá tối đa {self.max_price:,} VND "
                f"cho hành khách {self.passenger_name}.")

    def is_ok(self, flight: dict) -> bool:
        # kiểm cả chặng, ngày, giờ và kiểu dữ liệu. Ghế trống là
        # trạng thái tài nguyên, không phải điều kiện của booking đã giữ xong.
        if not isinstance(flight, dict):
            return False
        try:
            depart = datetime.fromisoformat(flight["depart"])
            price = flight["price"]
            valid = (flight.get("origin") == self.origin
                     and flight.get("destination") == self.destination
                     and depart.date() == Date.fromisoformat(self.date)
                     and depart.time() < time.fromisoformat(self.depart_before)
                     and type(price) is int and 0 <= price <= self.max_price)
            if "passenger" in flight:
                valid = valid and flight["passenger"] == self.passenger_name
            return valid
        except (KeyError, TypeError, ValueError):
            return False


def _flight(code: str, depart: str, price: int, seats: int) -> dict:
    """Dữ liệu mock đầy đủ chặng, không suy ra chặng từ prompt của model."""
    return {"flight": code, "origin": "SGN", "destination": "DAD",
            "depart": f"2026-10-07T{depart}", "price": price, "seats_available": seats}


class FlightDatabase:
    """Database mô phỏng; mỗi lượt benchmark có dữ liệu độc lập."""

    DEFAULT_FLIGHTS = {
        "success": [_flight("VN122", "08:10", 1_850_000, 5), _flight("QH118", "15:40", 1_640_000, 9)],
        "fail": [_flight("VJ604", "08:10", 2_480_000, 3), _flight("QH118", "15:40", 1_640_000, 9)],
        # search thấy còn ghế; ghế mất ngay trước book. Đây là
        # biến động giữa hai tool calls, không phải cố chọn chuyến đã biết hết ghế.
        "contingency": [_flight("VN122", "08:10", 1_850_000, 5),
                        _flight("VN124", "10:15", 1_950_000, 4), _flight("QH118", "15:40", 1_640_000, 9)],
        "transient_error": [_flight("VN122", "08:10", 1_850_000, 5)],
    }

    def __init__(self, scenario: str = "success"):
        self.scenario = scenario
        self.bookings: dict[str, dict] = {}
        self.reset(scenario)

    def reset(self, scenario: Optional[str] = None):
        selected = scenario or self.scenario
        if selected not in self.DEFAULT_FLIGHTS:
            raise ValueError(f"Kịch bản không tồn tại: {selected}")
        self.scenario = selected
        self.flights = deepcopy(self.DEFAULT_FLIGHTS[selected])
        self.bookings.clear()
        self.transient_error_counter = 0
        self._reservation_race_triggered = False
        self._booking_sequence = 0

    def search_flights(self, origin: str, destination: str, date: str) -> dict:
        if self.scenario == "transient_error" and self.transient_error_counter == 0:
            self.transient_error_counter += 1
            return {"status": "error", "error_code": "TRANSIENT_ERROR",
                    "retryable": True, "error": "Dịch vụ tạm thời gián đoạn."}
        # lọc đúng tham số, không trả toàn bộ scenario.
        return {"status": "ok", "flights": [deepcopy(f) for f in self.flights
                if f["origin"] == origin and f["destination"] == destination and f["depart"][:10] == date]}

    def book_seat(self, flight: str, passenger_name: str = "Nguyen Van A") -> dict:
        info = next((f for f in self.flights if f["flight"] == flight), None)
        if info is None:
            return {"status": "not_found", "error_code": "INVALID_FLIGHT",
                    "retryable": False, "error": f"Không có chuyến {flight}."}
        if self.scenario == "contingency" and flight == "VN122" and not self._reservation_race_triggered:
            self._reservation_race_triggered = True
            info["seats_available"] = 0
        if info["seats_available"] <= 0:
            return {"status": "error", "flight": flight, "error_code": "SOLD_OUT",
                    "retryable": False, "error": f"Chuyến {flight} đã hết chỗ."}
        info["seats_available"] -= 1
        # mã duy nhất, tránh ghi đè booking và làm mất trạng thái ghế.
        self._booking_sequence += 1
        code = f"{flight}-{self._booking_sequence:04d}"
        record = {"code": code, "flight": flight, "passenger": passenger_name,
                  "origin": info["origin"], "destination": info["destination"],
                  "depart": info["depart"], "price": info["price"],
                  "paid": False, "booking_status": "held"}
        self.bookings[code] = record
        return {"status": "ok", **deepcopy(record)}

    def pay(self, code: str, payment_method: str = "wallet") -> dict:
        """Thanh toán mock: held -> confirmed, không kết nối ví thật."""
        booking = self.bookings.get(code)
        if booking is None:
            return {"status": "not_found", "error": "Mã booking không tồn tại."}
        # DB cũng kiểm trạng thái nếu bị gọi trực tiếp. Booking
        # cancelled/confirmed không được thanh toán lại hoặc hồi sinh.
        if booking["booking_status"] != "held" or booking["paid"]:
            return {"status": "error", "error_code": "INVALID_BOOKING_STATE",
                    "retryable": False, "error": "Chỉ thanh toán booking đang held."}
        booking.update(paid=True, booking_status="confirmed", payment_method=payment_method)
        return {"status": "ok", **deepcopy(booking)}

    def get_booking(self, code: str) -> dict:
        return {"status": "ok", **deepcopy(self.bookings[code])} if code in self.bookings else {"status": "not_found", "code": code}

    def cancel_booking(self, code: str) -> dict:
        booking = self.bookings.get(code)
        if booking is None:
            return {"status": "not_found", "code": code}
        # Mock chưa có hoàn tiền, nên không tự ý hủy vé đã thanh toán.
        if booking["paid"]:
            return {"status": "error", "error_code": "REFUND_REQUIRES_HUMAN",
                    "retryable": False, "error": "Vé đã trả tiền cần người xử lý hoàn tiền."}
        if booking["booking_status"] != "cancelled":
            info = next(f for f in self.flights if f["flight"] == booking["flight"])
            info["seats_available"] += 1
            booking["booking_status"] = "cancelled"
        return {"status": "ok", **deepcopy(booking)}


class StopAgent(Exception):
    """Dừng vòng model/tool khi Harness phát hiện loop hoặc vượt ngân sách."""


class HarnessGuard:
    ALLOWED_TOOLS = {"search_flights", "book_seat", "pay", "get_booking", "cancel_booking"}

    def __init__(self, constraints: FlightConstraints, db: FlightDatabase):
        self.constraints = constraints
        self.db = db
        self.log: list[tuple[str, dict, dict]] = []
        self.stop_reason: Optional[str] = None
        self._tool_lock = RLock()

    def reset(self):
        self.log.clear()
        self.stop_reason = None

    def check_permission(self, tool_name: str, args: dict) -> Optional[str]:
        # allowlist ngăn gọi nhầm method nội bộ như reset.
        if not isinstance(args, dict):
            return "BLOCKED [INVALID_ARGUMENTS]: Tham số tool phải là dictionary."
        if tool_name not in self.ALLOWED_TOOLS:
            return f"BLOCKED [UNKNOWN_TOOL]: {tool_name} không được cấp quyền."
        if self.stop_reason:
            return self.stop_reason
        if self.is_looping(tool_name, args):
            self.stop_reason = f"BLOCKED [INFINITE_LOOP]: {tool_name} đã gọi cùng tham số 3 lần."
            return self.stop_reason
        if tool_name == "book_seat":
            info = next((f for f in self.db.flights if f["flight"] == args.get("flight")), None)
            if info is None:
                return "BLOCKED [INVALID_FLIGHT]: Chuyến bay không tồn tại."
            if not self.constraints.is_ok(info):
                return f"BLOCKED [GOAL_DRIFT]: Chuyến bay vi phạm {self.constraints.to_prompt()}"
            if args.get("passenger_name", "Nguyen Van A") != self.constraints.passenger_name:
                return "BLOCKED [WRONG_PASSENGER]: Hành khách khác yêu cầu."
            if any(b.get("passenger") == self.constraints.passenger_name and b.get("booking_status") in {"held", "confirmed"}
                   for b in self.db.bookings.values()):
                return "BLOCKED [DUPLICATE_BOOKING]: Hành khách đã có booking đang hiệu lực."
        if tool_name in {"pay", "get_booking", "cancel_booking"}:
            code = args.get("code")
            if not isinstance(code, str) or not code or code.startswith("$"):
                return "BLOCKED [INVALID_CODE]: Mã booking chưa được resolve hợp lệ."
            booking = self.db.bookings.get(code)
            if booking is None:
                return "BLOCKED [UNAUTHORIZED_BOOKING]: Mã booking không tồn tại."
            if booking.get("passenger") != self.constraints.passenger_name:
                return "BLOCKED [WRONG_PASSENGER]: Không được thao tác booking của người khác."
            if tool_name == "pay":
                # mã tồn tại chưa đủ; phải held, chưa trả tiền và
                # đạt ràng buộc trước khi thực hiện thanh toán.
                if booking.get("booking_status") != "held" or booking.get("paid") is not False:
                    return "BLOCKED [INVALID_BOOKING_STATE]: Booking không ở trạng thái held."
                if not self.constraints.is_ok(booking):
                    return "BLOCKED [GOAL_DRIFT]: Booking không còn đạt ràng buộc."
        return None

    def is_looping(self, tool_name: str, args: dict) -> bool:
        return sum(t == tool_name and a == args for t, a, _ in self.log) >= 3

    def normalize_result(self, tool_name: str, result) -> dict:
        # {} / thiếu status / search thiếu flights là lỗi dữ liệu,
        # không phải 'không có chuyến'. retryable giúp agent phục hồi có giới hạn.
        invalid = not isinstance(result, dict) or result.get("status") not in {"ok", "error", "denied", "not_found", "warning"}
        if not invalid and tool_name == "search_flights" and result["status"] == "ok":
            invalid = not isinstance(result.get("flights"), list)
        if not invalid and tool_name in {"book_seat", "pay", "get_booking"} and result["status"] == "ok":
            invalid = not {"code", "flight", "origin", "destination", "depart", "price", "passenger", "paid", "booking_status"}.issubset(result)
        if invalid:
            return {"status": "error", "error_code": "STATE_CORRUPTION",
                    "retryable": True, "error": "Tool trả dữ liệu rỗng hoặc sai cấu trúc."}
        normalized = deepcopy(result)
        if normalized["status"] != "ok":
            normalized.setdefault("retryable", False)
        return normalized

    def execute_tool(self, tool_name: str, args: dict) -> dict:
        # NOTE API 11: ReAct/LLM thật có thể sinh nhiều tool calls đồng thời.
        # Khóa bao trọn kiểm quyền -> ghi DB -> log để hai lệnh book không cùng
        # vượt qua check khi chưa thấy booking của lệnh kia. Không khóa model/API.
        with self._tool_lock:
            return self._execute_locked_tool(tool_name, args)

    def _execute_locked_tool(self, tool_name: str, args: dict) -> dict:
        reason = self.check_permission(tool_name, args)
        if reason:
            result = {"status": "denied", "reason": reason, "retryable": False}
        else:
            try:
                result = self.normalize_result(tool_name, getattr(self.db, tool_name)(**args))
            except Exception as exc:
                result = {"status": "error", "error_code": "TOOL_EXCEPTION",
                          "retryable": isinstance(exc, (TimeoutError, ConnectionError)), "error": str(exc)}
        # Snapshot tránh thao tác DB sau đó sửa ngược lịch sử quan sát.
        self.log.append((tool_name, deepcopy(args), deepcopy(result)))
        return result

    def check_hallucination(self, final_answer: str) -> list[str]:
        # chỉ dùng tool thành công làm căn cứ; kiểm cả mã và giá.
        # Đây là kiểm tra hữu hạn cho mã/giá, không chứng minh mọi câu đều đúng.
        facts: set[tuple[str, int]] = set()
        for _, _, result in self.log:
            if result.get("status") != "ok":
                continue
            flights = result.get("flights", [])
            for row in (flights if isinstance(flights, list) else []) + [result]:
                if isinstance(row, dict) and isinstance(row.get("flight"), str) and type(row.get("price")) is int:
                    facts.add((row["flight"], row["price"]))
        mentions = list(re.finditer(r"\b[A-Z]{2}\d{3}\b", final_answer))
        known = {code for code, _ in facts}
        ungrounded = [m.group() for m in mentions if m.group() not in known]
        money_pattern = r"(?<!\d)\d{1,3}(?:[., ]\d{3})+(?!\d)|\b\d{4,}(?=\s*(?:VND|VNĐ|đ))"
        for match in re.finditer(money_pattern, final_answer, flags=re.IGNORECASE):
            price = int(re.sub(r"[., ]", "", match.group()))
            preceding = [m for m in mentions if m.start() < match.start()]
            code = preceding[-1].group() if preceding else None
            grounded = (code, price) in facts if code else any(p == price for _, p in facts)
            if not grounded:
                ungrounded.append(str(price))
        return list(dict.fromkeys(ungrounded))

    def is_done(self) -> bool:
        # úng người + confirmed + paid + chặng/ngày/giờ/giá.
        # Không tin vào câu 'đã xong' của model.
        for code in self.db.bookings:
            current = self.db.get_booking(code)
            if (current.get("status") == "ok" and current.get("paid") is True
                    and current.get("booking_status") == "confirmed"
                    and current.get("passenger") == self.constraints.passenger_name
                    and self.constraints.is_ok(current)):
                return True
        return False

    def handoff(self, custom_question: Optional[str] = None) -> dict:
        # booking_status thuộc DB; status thuộc response tool.
        # .get() vẫn cho phép bàn giao khi dữ liệu đang thiếu trường.
        done_so_far = [f"Mã {code}: chuyến {b.get('flight', '?')}, đã thanh toán={b.get('paid', False)}, "
                       f"trạng thái={b.get('booking_status', 'unknown')}" for code, b in self.db.bookings.items()]
        return {
            "done_so_far": done_so_far or ["Chưa có chỗ nào được giữ hoặc thanh toán."],
            "tried": [f"{t}({a}) -> {r.get('status')}: {r.get('error', r.get('reason', 'OK'))}" for t, a, r in self.log],
            "question": custom_question or (f"Chưa hoàn tất yêu cầu: {self.constraints.to_prompt()} "
                                             "Bạn muốn đổi giới hạn giờ bay, ngân sách hay nhờ người xử lý tiếp?"),
        }


def create_langchain_tools(guard: HarnessGuard):
    """Mọi wrapper đều gọi Harness trước khi chạm database mô phỏng."""

    @tool
    def search_flights(origin: str, destination: str, date: str) -> dict:
        """Tìm chuyến bay theo chặng và ngày YYYY-MM-DD."""
        return guard.execute_tool("search_flights", {"origin": origin, "destination": destination, "date": date})

    @tool
    def book_seat(flight: str, passenger_name: str = "Nguyen Van A") -> dict:
        """Giữ một chỗ, chưa thanh toán; trả mã booking duy nhất."""
        return guard.execute_tool("book_seat", {"flight": flight, "passenger_name": passenger_name})

    @tool
    def pay(code: str, payment_method: str = "wallet") -> dict:
        """Thanh toán mô phỏng một booking đang held."""
        return guard.execute_tool("pay", {"code": code, "payment_method": payment_method})

    @tool
    def get_booking(code: str) -> dict:
        """Đọc booking từ database mô phỏng."""
        return guard.execute_tool("get_booking", {"code": code})

    @tool
    def cancel_booking(code: str) -> dict:
        """Hủy giữ chỗ chưa trả tiền; vé đã trả tiền cần người xử lý."""
        return guard.execute_tool("cancel_booking", {"code": code})

    return [search_flights, book_seat, pay, get_booking, cancel_booking]
