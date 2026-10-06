"""Kiểm thử hồi quy cho các lỗi nghiệp vụ đã xác nhận trong lần rà soát.

Chạy: python -m unittest -v test_agents
NOTE: Chỉ dùng database/model mock; không cần API key hoặc mạng.
"""

from __future__ import annotations

import contextlib
import io
import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from time import sleep
from unittest.mock import patch

from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from agents import HybridFlightAgent, Plan, PlanThenExecuteFlightAgent, ReActFlightAgent
from evaluate_agents import aggregate_metrics, run_benchmark
from flight_harness import FlightConstraints, FlightDatabase, HarnessGuard


class ToolCallingFakeModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class PaymentFailureDatabase(FlightDatabase):
    def pay(self, code, payment_method="wallet"):
        return {"status": "error", "error_code": "PAYMENT_UNCERTAIN", "retryable": False,
                "error": "Dịch vụ thanh toán chưa xác nhận trạng thái."}


class EmptyOnceDatabase(FlightDatabase):
    def search_flights(self, origin, destination, date):
        if not getattr(self, "empty_returned", False):
            self.empty_returned = True
            return {}
        return super().search_flights(origin, destination, date)


class PermanentSearchFailureDatabase(FlightDatabase):
    def search_flights(self, origin, destination, date):
        return {"status": "error", "error_code": "INVALID_REQUEST", "retryable": False,
                "error": "Lỗi không thể phục hồi bằng retry."}


class FlightTestCase(unittest.TestCase):
    def make(self, scenario="success", db_type=FlightDatabase):
        constraints = FlightConstraints()
        db = db_type(scenario)
        guard = HarnessGuard(constraints, db)
        return constraints, db, guard

    def search_args(self, constraints):
        return {"origin": constraints.origin, "destination": constraints.destination, "date": constraints.date}


class HarnessTests(FlightTestCase):
    def test_parallel_tool_calls_cannot_duplicate_booking(self):
        # LLM thật có thể đề xuất nhiều book calls trong một response. Trì hoãn
        # thao tác DB để kiểm khoảng trống giữa check quyền và tạo booking.
        class SlowBookingDatabase(FlightDatabase):
            def book_seat(self, flight, passenger_name="Nguyen Van A"):
                sleep(0.02)
                return super().book_seat(flight, passenger_name)
        c, db, guard = self.make(db_type=SlowBookingDatabase)
        with ThreadPoolExecutor(max_workers=5) as pool:
            results = list(pool.map(lambda _: guard.execute_tool("book_seat", {
                "flight": "VN122", "passenger_name": c.passenger_name}), range(5)))
        self.assertEqual(sum(r["status"] == "ok" for r in results), 1)
        self.assertEqual(len(db.bookings), 1)
        self.assertEqual(db.flights[0]["seats_available"], 4)

    def test_route_date_time_and_price_fail_closed(self):
        c, db, _ = self.make()
        flight = db.flights[0]
        self.assertTrue(c.is_ok(flight))
        for change in [{"origin": "HAN"}, {"destination": "CXR"}, {"depart": "2026-10-08T08:10"},
                       {"depart": "2026-10-07T12:00"}, {"depart": None}, {"price": 2_000_001},
                       {"price": -1}, {"price": "1850000"}, {"price": True}, {"passenger": "Wrong"}]:
            with self.subTest(change=change):
                self.assertFalse(c.is_ok(dict(flight, **change)))
        self.assertTrue(c.is_ok(dict(flight, price=2_000_000)))

    def test_search_filters_route_and_date(self):
        _, db, _ = self.make()
        self.assertEqual(db.search_flights("HAN", "CXR", "2026-10-07")["flights"], [])
        self.assertEqual(db.search_flights("SGN", "DAD", "2099-01-01")["flights"], [])

    def test_wrong_passenger_denied_before_booking(self):
        _, db, guard = self.make()
        self.assertEqual(guard.execute_tool("book_seat", {"flight": "VN122", "passenger_name": "Wrong"})["status"], "denied")
        self.assertFalse(db.bookings)
        self.assertEqual(db.flights[0]["seats_available"], 5)

    def test_wrong_passenger_cannot_pay_or_complete(self):
        _, db, guard = self.make()
        code = db.book_seat("VN122", "Wrong")["code"]
        self.assertEqual(guard.execute_tool("pay", {"code": code})["status"], "denied")
        db.pay(code)
        self.assertFalse(guard.is_done())

    def test_handoff_retains_held_booking_state(self):
        _, _, guard = self.make()
        code = guard.execute_tool("book_seat", {"flight": "VN122"})["code"]
        handoff = guard.handoff()
        self.assertEqual(set(handoff), {"done_so_far", "tried", "question"})
        self.assertIn(code, handoff["done_so_far"][0])
        self.assertIn("held", handoff["done_so_far"][0])
        self.assertFalse(guard.is_done())

    def test_cancelled_booking_cannot_be_paid_or_revived(self):
        _, db, guard = self.make()
        code = guard.execute_tool("book_seat", {"flight": "VN122"})["code"]
        guard.execute_tool("cancel_booking", {"code": code})
        self.assertEqual(db.flights[0]["seats_available"], 5)
        self.assertEqual(guard.execute_tool("pay", {"code": code})["status"], "denied")
        self.assertEqual(db.pay(code)["status"], "error")
        self.assertEqual(db.bookings[code]["booking_status"], "cancelled")
        self.assertFalse(guard.is_done())

    def test_constraint_change_in_booking_blocks_payment(self):
        _, db, guard = self.make()
        code = guard.execute_tool("book_seat", {"flight": "VN122"})["code"]
        db.bookings[code]["price"] = 9_000_000
        self.assertEqual(guard.execute_tool("pay", {"code": code})["status"], "denied")
        self.assertFalse(db.bookings[code]["paid"])

    def test_no_duplicate_booking_or_payment(self):
        _, db, guard = self.make()
        code = guard.execute_tool("book_seat", {"flight": "VN122"})["code"]
        self.assertEqual(guard.execute_tool("book_seat", {"flight": "VN122"})["status"], "denied")
        self.assertEqual(guard.execute_tool("pay", {"code": code})["status"], "ok")
        self.assertEqual(guard.execute_tool("pay", {"code": code})["status"], "denied")
        self.assertEqual(len(db.bookings), 1)
        self.assertTrue(guard.is_done())

    def test_booking_codes_do_not_overwrite(self):
        _, db, _ = self.make()
        first = db.book_seat("VN122", "A")["code"]
        second = db.book_seat("VN122", "B")["code"]
        self.assertNotEqual(first, second)
        self.assertEqual(len(db.bookings), 2)

    def test_unknown_tools_and_placeholders_denied(self):
        _, db, guard = self.make()
        for name, args in [("reset", {}), ("pay", {"code": "$booking_code"}), ("pay", {"code": "UNKNOWN"})]:
            self.assertEqual(guard.execute_tool(name, args)["status"], "denied")
        self.assertFalse(db.bookings)

    def test_loop_stops_after_three_actual_calls(self):
        c, _, guard = self.make()
        statuses = [guard.execute_tool("search_flights", self.search_args(c))["status"] for _ in range(4)]
        self.assertEqual(statuses, ["ok", "ok", "ok", "denied"])
        self.assertIn("INFINITE_LOOP", guard.stop_reason)

    def test_corrupt_tool_result_is_retryable_error(self):
        _, _, guard = self.make()
        for value in [{}, None, {"flights": []}, {"status": "ok"}, {"status": "unknown"}]:
            with self.subTest(value=value):
                result = guard.normalize_result("search_flights", value)
                self.assertEqual(result["error_code"], "STATE_CORRUPTION")
                self.assertTrue(result["retryable"])
        self.assertEqual(guard.normalize_result("search_flights", {"status": "ok", "flights": []})["status"], "ok")

    def test_hallucination_checks_prices_and_flight_price_pair(self):
        c, _, guard = self.make("contingency")
        guard.execute_tool("search_flights", self.search_args(c))
        self.assertIn("VN999", guard.check_hallucination("VN999 giá 1,200,000 VND."))
        self.assertIn("1200000", guard.check_hallucination("VN122 giá 1,200,000 VND."))
        self.assertIn("1950000", guard.check_hallucination("VN122 giá 1,950,000 VND."))
        for text in ["VN122 giá 1,850,000 VND.", "VN122 giá 1.850.000 VND.", "VN122 giá 1850000 VND."]:
            self.assertEqual(guard.check_hallucination(text), [])

    def test_sold_out_happens_after_search(self):
        c, db, guard = self.make("contingency")
        before = guard.execute_tool("search_flights", self.search_args(c))
        self.assertGreater(before["flights"][0]["seats_available"], 0)
        self.assertEqual(guard.execute_tool("book_seat", {"flight": "VN122"})["error_code"], "SOLD_OUT")
        self.assertEqual(db.flights[0]["seats_available"], 0)
        self.assertGreater(guard.log[0][2]["flights"][0]["seats_available"], 0)


class AgentTests(FlightTestCase):
    def test_all_three_patterns_use_actual_model_callbacks(self):
        for agent_type in [ReActFlightAgent, PlanThenExecuteFlightAgent, HybridFlightAgent]:
            with self.subTest(agent=agent_type.__name__):
                c, db, guard = self.make()
                agent = agent_type(c, db, guard)
                result = agent.run()
                self.assertEqual(result.status, "SUCCESS")
                self.assertEqual(result.llm_calls, sum(t.get("step") == "model_call" for t in result.trace))
                self.assertEqual(result.tool_calls, len(guard.log))
                self.assertTrue(guard.is_done())
                if agent_type != PlanThenExecuteFlightAgent:
                    self.assertEqual(type(agent.runtime).__name__, "CompiledStateGraph")

    def test_pending_review_cannot_write_for_plan_or_hybrid(self):
        for agent_type in [PlanThenExecuteFlightAgent, HybridFlightAgent]:
            with self.subTest(agent=agent_type.__name__):
                c, db, guard = self.make()
                result = agent_type(c, db, guard).run(auto_approve=False)
                self.assertEqual(result.status, "PENDING_REVIEW")
                self.assertFalse(db.bookings)
                self.assertEqual([t for t, _, _ in guard.log], ["search_flights"])

    def test_rejected_plan_cannot_write(self):
        for agent_type in [PlanThenExecuteFlightAgent, HybridFlightAgent]:
            c, db, guard = self.make()
            result = agent_type(c, db, guard).run(reviewer=lambda plan: False)
            self.assertEqual(result.status, "HANDOFF")
            self.assertFalse(db.bookings)

    def test_prepared_plan_cannot_change_after_review(self):
        c, db, guard = self.make()
        agent = PlanThenExecuteFlightAgent(c, db, guard)
        plan = agent.prepare_plan()
        plan.steps[0].args["flight"] = "QH118"
        self.assertEqual(agent.execute_plan(plan, approved=True).status, "HANDOFF")
        self.assertFalse(db.bookings)

    def test_same_prepared_plan_can_be_approved_once(self):
        c, db, guard = self.make()
        agent = PlanThenExecuteFlightAgent(c, db, guard)
        plan = agent.prepare_plan()
        self.assertEqual(agent.execute_plan(plan).status, "PENDING_REVIEW")
        self.assertEqual(agent.execute_plan(plan, approved=True).status, "SUCCESS")
        self.assertEqual(agent.execute_plan(plan, approved=True).status, "HANDOFF")
        self.assertEqual(len(db.bookings), 1)
        self.assertEqual(sum(t == "search_flights" for t, _, _ in guard.log), 1)

    def test_hybrid_replan_is_reviewed_before_new_booking(self):
        c, db, guard = self.make("contingency")
        seen = []
        def reviewer(plan):
            seen.append(plan.steps[0].args["flight"])
            return len(seen) == 1
        result = HybridFlightAgent(c, db, guard).run(reviewer=reviewer)
        self.assertEqual(seen, ["VN122", "VN124"])
        self.assertEqual(result.status, "HANDOFF")
        self.assertFalse(db.bookings)
        self.assertEqual([a["flight"] for t, a, _ in guard.log if t == "book_seat"], ["VN122"])

    def test_hybrid_replan_budget_is_enforced(self):
        c, db, guard = self.make("contingency")
        agent = HybridFlightAgent(c, db, guard)
        agent.max_replans = 0
        result = agent.run()
        self.assertEqual(result.status, "HANDOFF")
        self.assertEqual(result.llm_calls, 1)
        self.assertFalse(db.bookings)

    def test_search_error_does_not_generate_or_approve_empty_data(self):
        c, db, guard = self.make("transient_error")
        reviewer = unittest.mock.Mock(return_value=True)
        result = PlanThenExecuteFlightAgent(c, db, guard).run(reviewer=reviewer)
        self.assertEqual(result.status, "HANDOFF")
        self.assertEqual(result.llm_calls, 0)
        reviewer.assert_not_called()
        self.assertEqual(len(guard.log), 1)
        self.assertFalse(db.bookings)

    def test_cli_plan_transient_has_no_second_search_or_payment(self):
        import main
        with patch.object(main, "print_result") as show, patch("builtins.input", side_effect=["4"]):
            main.run_plan_then_execute_interactive()
        result = show.call_args.args[0]
        self.assertEqual(result.status, "HANDOFF")
        self.assertEqual(result.tool_calls, 1)
        self.assertFalse(result.output_details["bookings"])

    def test_payment_failure_handoffs_with_held_booking(self):
        for agent_type in [ReActFlightAgent, PlanThenExecuteFlightAgent, HybridFlightAgent]:
            with self.subTest(agent=agent_type.__name__):
                c, db, guard = self.make(db_type=PaymentFailureDatabase)
                result = agent_type(c, db, guard).run()
                self.assertEqual(result.status, "HANDOFF")
                self.assertFalse(result.is_done)
                self.assertEqual(len(db.bookings), 1)
                self.assertIn("held", result.output_details["handoff"]["done_so_far"][0])
                self.assertEqual(sum(t == "pay" for t, _, _ in guard.log), 1)

    def test_corrupt_search_can_recover_in_react_and_hybrid(self):
        for agent_type in [ReActFlightAgent, HybridFlightAgent]:
            c, db, guard = self.make(db_type=EmptyOnceDatabase)
            result = agent_type(c, db, guard).run()
            self.assertEqual(result.status, "SUCCESS")
            self.assertEqual(guard.log[0][2]["error_code"], "STATE_CORRUPTION")
            self.assertEqual(sum(t == "search_flights" for t, _, _ in guard.log), 2)

    def test_permanent_search_errors_are_not_retried(self):
        for agent_type in [ReActFlightAgent, PlanThenExecuteFlightAgent, HybridFlightAgent]:
            c, _, guard = self.make(db_type=PermanentSearchFailureDatabase)
            self.assertEqual(agent_type(c, guard.db, guard).run().status, "HANDOFF")
            self.assertEqual(len(guard.log), 1)

    def test_react_model_budget_and_loop_stop_are_enforced(self):
        for budget in [2, 10]:
            c, db, guard = self.make()
            messages = [AIMessage(content="", tool_calls=[{"name": "search_flights", "args": self.search_args(c), "id": f"s{i}"}]) for i in range(15)]
            agent = ReActFlightAgent(c, db, guard)
            agent.max_steps = budget
            result = agent.run(fake_mode=False, llm=ToolCallingFakeModel(messages=iter(messages)))
            self.assertEqual(result.status, "HANDOFF")
            self.assertLessEqual(result.llm_calls, budget)
            self.assertLessEqual(len(guard.log), 4)
            self.assertFalse(db.bookings)
            if budget == 10:
                self.assertIn("INFINITE_LOOP", guard.stop_reason)

    def test_react_final_hallucination_is_checked(self):
        c, db, guard = self.make()
        search = AIMessage(content="", tool_calls=[{"name": "search_flights", "args": self.search_args(c), "id": "search"}])
        model = ToolCallingFakeModel(messages=iter([search, AIMessage(content="Đã đặt VN999 giá 1,200,000 VND.")]))
        result = ReActFlightAgent(c, db, guard).run(fake_mode=False, llm=model)
        self.assertEqual(result.status, "HANDOFF")
        self.assertIn("VN999", result.output_details["hallucinations"])
        self.assertFalse(db.bookings)

    def test_provided_planner_model_is_actually_invoked(self):
        c, db, guard = self.make()
        plan = Plan.model_validate({"steps": [{"tool": "book_seat", "args": {"flight": "VN122", "passenger_name": c.passenger_name}},
                                               {"tool": "pay", "args": {"code": "$booking_code"}}]})
        class ProvidedPlanner(GenericFakeChatModel):
            def with_structured_output(self, schema, **kwargs):
                from langchain_core.output_parsers import PydanticOutputParser
                return self | PydanticOutputParser(pydantic_object=schema)
        model = ProvidedPlanner(messages=iter([AIMessage(content=plan.model_dump_json())]))
        result = PlanThenExecuteFlightAgent(c, db, guard).run(fake_mode=False, llm=model)
        self.assertEqual(result.status, "SUCCESS")
        self.assertEqual(result.llm_calls, 1)
        self.assertEqual(result.model_mode, "provided_model")


class BenchmarkTests(unittest.TestCase):
    def test_expected_outcomes_and_metrics_on_all_twelve_cases(self):
        with contextlib.redirect_stdout(io.StringIO()):
            results = run_benchmark()
        self.assertEqual(len(results), 12)
        metrics = {m["pattern"]: m for m in aggregate_metrics(results)}
        self.assertEqual(metrics["ReAct"]["correct_behavior"], 4)
        self.assertEqual(metrics["Plan-then-Execute"]["correct_behavior"], 2)
        self.assertEqual(metrics["Lai (Hybrid)"]["correct_behavior"], 4)
        self.assertEqual(metrics["ReAct"]["avg_model_calls"], 4.0)
        self.assertEqual(metrics["Plan-then-Execute"]["avg_model_calls"], 0.75)
        self.assertEqual(metrics["Lai (Hybrid)"]["avg_model_calls"], 1.25)
        for metric in metrics.values():
            self.assertEqual(metric["invalid_active_bookings"], 0)


if __name__ == "__main__":
    unittest.main()
