import pytest
from learnm8.utils.cycle_utils import parse_cycle_spec, summarize_cycle_spec, validate_cycle_spec


@pytest.mark.unit
class TestParseCycleSpec:

    def test_single_strategy(self):
        result = parse_cycle_spec("random:0.01")
        assert result == [('random', 0.01)]

    def test_multiple_strategies(self):
        result = parse_cycle_spec("random:0.01 greedy:0.005")
        assert result == [('random', 0.01), ('greedy', 0.005)]

    def test_repeat_with_multiplier(self):
        result = parse_cycle_spec("greedy:0.005*3")
        assert result == [('greedy', 0.005), ('greedy', 0.005), ('greedy', 0.005)]

    def test_mixed_with_and_without_multiplier(self):
        result = parse_cycle_spec("random:0.01 greedy:0.005*5 ucb:0.01")
        assert len(result) == 7
        assert result[0] == ('random', 0.01)
        assert result[1:6] == [('greedy', 0.005)] * 5
        assert result[6] == ('ucb', 0.01)

    def test_multiplier_of_one(self):
        result = parse_cycle_spec("random:0.01*1")
        assert result == [('random', 0.01)]

    def test_various_strategies(self):
        result = parse_cycle_spec("ucb:0.02 ei:0.01 pi:0.015")
        assert result == [('ucb', 0.02), ('ei', 0.01), ('pi', 0.015)]

    def test_fraction_precision(self):
        result = parse_cycle_spec("greedy:0.001")
        assert result[0][1] == pytest.approx(0.001)

    def test_docstring_example(self):
        result = parse_cycle_spec("random:0.01 greedy:0.005*5")
        assert result == [
            ('random', 0.01),
            ('greedy', 0.005), ('greedy', 0.005), ('greedy', 0.005),
            ('greedy', 0.005), ('greedy', 0.005)
        ]


@pytest.mark.unit
class TestSummarizeCycleSpec:

    def test_single_strategy(self):
        result = summarize_cycle_spec("random:0.01")
        assert result == "r1"

    def test_multiple_strategies(self):
        result = summarize_cycle_spec("random:0.01 greedy:0.005")
        assert result == "r1_g1"

    def test_with_multiplier(self):
        result = summarize_cycle_spec("greedy:0.005*5")
        assert result == "g5"

    def test_mixed_spec(self):
        result = summarize_cycle_spec("random:0.01 greedy:0.005*3 ucb:0.01")
        assert result == "r1_g3_u1"

    def test_all_known_strategy_abbreviations(self):
        result = summarize_cycle_spec("random:0.01 greedy:0.01 ucb:0.01 ei:0.01 pi:0.01 thompson:0.01 entropy:0.01 bitbirch:0.01 simulated_annealing:0.01")
        assert result == "r1_g1_u1_e1_p1_t1_h1_b1_s1"

    def test_unknown_strategy_uses_first_char(self):
        result = summarize_cycle_spec("custom_strat:0.01")
        assert result == "c1"

    def test_multiplier_preserved_in_summary(self):
        result = summarize_cycle_spec("random:0.01*10")
        assert result == "r10"


@pytest.mark.unit
class TestValidateCycleSpec:

    def test_valid_simple_spec(self):
        is_valid, msg = validate_cycle_spec("random:0.01")
        assert is_valid is True
        assert msg == ""

    def test_valid_complex_spec(self):
        is_valid, msg = validate_cycle_spec("random:0.01 greedy:0.005*5 ucb:0.01")
        assert is_valid is True
        assert msg == ""

    def test_invalid_fraction_zero(self):
        is_valid, msg = validate_cycle_spec("random:0.0")
        assert is_valid is False
        assert "Invalid batch fraction" in msg

    def test_invalid_fraction_above_one(self):
        is_valid, msg = validate_cycle_spec("random:1.5")
        assert is_valid is False
        assert "Invalid batch fraction" in msg

    def test_fraction_exactly_one_is_valid(self):
        is_valid, msg = validate_cycle_spec("random:1.0")
        assert is_valid is True

    def test_malformed_spec_no_colon(self):
        is_valid, msg = validate_cycle_spec("random0.01")
        assert is_valid is False
        assert msg != ""

    def test_malformed_spec_empty_string(self):
        is_valid, msg = validate_cycle_spec("")
        assert is_valid is False
        assert msg != ""

    def test_malformed_spec_missing_fraction(self):
        is_valid, msg = validate_cycle_spec("random:")
        assert is_valid is False
        assert msg != ""

    def test_returns_tuple(self):
        result = validate_cycle_spec("random:0.01")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)
