import unittest
from evaluate import normalize, score, token_f1


class EvaluationTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize('  Hello, WORLD!'), 'hello world')

    def test_token_f1(self):
        self.assertEqual(token_f1('a b', 'a b'), 1)
        self.assertEqual(token_f1('cat', 'dog'), 0)

    def test_scoring(self):
        case = [{'id': '1', 'question': 'Q?', 'reference_answer': 'yes',
                 'model_answer': 'yes', 'context': 'yes'}]
        report = score(case)
        self.assertEqual(report['exact_match_rate'], 1)
        self.assertTrue(report['examples'][0]['answer_in_context'])


if __name__ == '__main__':
    unittest.main()
