import unittest

from query_engine import QueryEngine
from query_operators import ANDNOTOperator, ANDOperator, OROperator, TermOperator
from query_parser import Operators


class TestQueryEngine(unittest.TestCase):
    def test_process_OR_query(self):
        engine = QueryEngine()
        root_operator = engine._process_phrase2query("a OR b")
        self.assertIsInstance(root_operator, OROperator)
        self.assertEqual(len(root_operator.children), 2)
        self.assertIsInstance(root_operator.children[0], TermOperator)
        self.assertIsInstance(root_operator.children[1], TermOperator)

    def test_process_AND_query(self):
        engine = QueryEngine()
        root_operator = engine._process_phrase2query("a AND a ANDNOT b")
        self.assertIsInstance(root_operator, ANDNOTOperator)
        self.assertEqual(len(root_operator.children), 2)
        self.assertIsInstance(root_operator.children[0], ANDOperator)
        self.assertIsInstance(root_operator.children[1], TermOperator)
        self.assertIsInstance(root_operator.children[0].children[0], TermOperator)
        self.assertIsInstance(root_operator.children[0].children[1], TermOperator)

    def test_operator_identification(self):
        operator_class = Operators.get_EnumOperator("AND")
        self.assertEqual(operator_class, Operators.AND) 
        operator_class = Operators.get_EnumOperator("or")
        self.assertEqual(operator_class, Operators.OR)
        operator_class = Operators.get_EnumOperator("andnot")
        self.assertEqual(operator_class, Operators.ANDNOT)
        operator_class = Operators.get_EnumOperator("c.a.t")
        self.assertEqual(operator_class, Operators.TERM)


    def test_DEMO_for_execution(self):
        engine = QueryEngine()
        root_operator = engine._process_phrase2query("banana AND banana OR cherry ANDNOT banana")
        result = root_operator.execute()
        expected_ids = {"ID-banana", "ID-cherry"}
        self.assertEqual(result, expected_ids)


if __name__ == '__main__':
    unittest.main()