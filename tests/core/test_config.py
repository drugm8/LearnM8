import pytest
from learnm8.core.config import CycleConfig, parse_cycle_spec, parse_cycle_schedule


@pytest.mark.unit
class TestCycleConfig:

    def test_valid_config_with_batch_fraction(self):
        config = CycleConfig('greedy', n_cycles=5, batch_fraction=0.01)

        assert config.strategy == 'greedy'
        assert config.n_cycles == 5
        assert config.batch_fraction == 0.01

    def test_config_with_pruning(self):
        config = CycleConfig(
            'greedy',
            n_cycles=3,
            batch_fraction=0.01,
            pruning_strategy='score',
            pruning_params={'pruning_fraction': 0.3}
        )

        assert config.pruning_strategy == 'score'
        assert config.pruning_params == {'pruning_fraction': 0.3}

    def test_config_with_acquisition_params(self):
        config = CycleConfig(
            'ucb',
            n_cycles=1,
            batch_fraction=0.01,
            acquisition_params={'exploration_weight': 2.0}
        )

        assert config.acquisition_params == {'exploration_weight': 2.0}

    def test_missing_batch_fraction_raises_error(self):
        with pytest.raises(ValueError, match="CycleConfig requires batch_fraction"):
            CycleConfig('greedy', n_cycles=1)

    def test_invalid_batch_fraction_zero(self):
        with pytest.raises(ValueError, match="batch_fraction must be between 0"):
            CycleConfig('greedy', n_cycles=1, batch_fraction=0)

    def test_invalid_batch_fraction_negative(self):
        with pytest.raises(ValueError, match="batch_fraction must be between 0"):
            CycleConfig('greedy', n_cycles=1, batch_fraction=-0.1)

    def test_invalid_batch_fraction_too_high(self):
        with pytest.raises(ValueError, match="batch_fraction must be between 0"):
            CycleConfig('greedy', n_cycles=1, batch_fraction=1.5)

    def test_valid_batch_fraction_edge_case_one(self):
        config = CycleConfig('greedy', n_cycles=1, batch_fraction=1.0)
        assert config.batch_fraction == 1.0

    def test_valid_batch_fraction_edge_case_small(self):
        config = CycleConfig('greedy', n_cycles=1, batch_fraction=0.0001)
        assert config.batch_fraction == 0.0001


@pytest.mark.unit
class TestParseCycleSpec:

    def test_single_cycle_spec(self):
        configs = parse_cycle_spec("random:0.02")

        assert len(configs) == 1
        assert configs[0]['strategy'] == 'random'
        assert configs[0]['batch_fraction'] == 0.02
        assert configs[0]['n_cycles'] == 1

    def test_single_cycle_with_count(self):
        configs = parse_cycle_spec("greedy:0.01*5")

        assert len(configs) == 1
        assert configs[0]['strategy'] == 'greedy'
        assert configs[0]['batch_fraction'] == 0.01
        assert configs[0]['n_cycles'] == 5

    def test_multiple_cycle_specs(self):
        configs = parse_cycle_spec("random:0.02 greedy:0.01*5 ucb:0.01*3")

        assert len(configs) == 3

        assert configs[0]['strategy'] == 'random'
        assert configs[0]['batch_fraction'] == 0.02
        assert configs[0]['n_cycles'] == 1

        assert configs[1]['strategy'] == 'greedy'
        assert configs[1]['batch_fraction'] == 0.01
        assert configs[1]['n_cycles'] == 5

        assert configs[2]['strategy'] == 'ucb'
        assert configs[2]['batch_fraction'] == 0.01
        assert configs[2]['n_cycles'] == 3

    def test_empty_spec_returns_empty_list(self):
        configs = parse_cycle_spec("")
        assert configs == []

    def test_whitespace_only_spec(self):
        configs = parse_cycle_spec("   ")
        assert configs == []

    def test_invalid_format_no_colon(self):
        with pytest.raises(ValueError, match="Invalid cycle specification"):
            parse_cycle_spec("greedy0.01")

    def test_invalid_batch_fraction_too_high(self):
        with pytest.raises(ValueError, match="batch_fraction must be between 0"):
            parse_cycle_spec("greedy:1.5")

    def test_invalid_batch_fraction_zero(self):
        with pytest.raises(ValueError, match="batch_fraction must be between 0"):
            parse_cycle_spec("greedy:0")

    def test_invalid_n_cycles_zero(self):
        with pytest.raises(ValueError, match="n_cycles must be >= 1"):
            parse_cycle_spec("greedy:0.01*0")

    def test_invalid_n_cycles_negative(self):
        with pytest.raises(ValueError, match="n_cycles must be >= 1"):
            parse_cycle_spec("greedy:0.01*-5")


@pytest.mark.unit
class TestParseCycleSchedule:

    def test_simple_api_default_parameters(self):
        schedule = parse_cycle_schedule(
            strategy='greedy',
            n_cycles=10,
            batch_fraction=0.01
        )

        assert len(schedule) == 10
        assert schedule[0].strategy == 'random'
        assert schedule[0].batch_fraction == 0.01

        for config in schedule[1:]:
            assert config.strategy == 'greedy'
            assert config.batch_fraction == 0.01
            assert config.n_cycles == 1

    def test_simple_api_custom_initial(self):
        schedule = parse_cycle_schedule(
            strategy='greedy',
            n_cycles=5,
            batch_fraction=0.01,
            initial_strategy='diverse'
        )

        assert len(schedule) == 5
        assert schedule[0].strategy == 'diverse'
        assert schedule[0].batch_fraction == 0.01
        for config in schedule[1:]:
            assert config.strategy == 'greedy'
            assert config.batch_fraction == 0.01

    def test_advanced_api_with_cycle_configs(self):
        cycles = [
            CycleConfig('random', n_cycles=3, batch_fraction=0.02),
            CycleConfig('greedy', n_cycles=5, batch_fraction=0.01)
        ]

        schedule = parse_cycle_schedule(cycles=cycles)

        assert len(schedule) == 8

        for i in range(3):
            assert schedule[i].strategy == 'random'
            assert schedule[i].batch_fraction == 0.02
            assert schedule[i].n_cycles == 1

        for i in range(3, 8):
            assert schedule[i].strategy == 'greedy'
            assert schedule[i].batch_fraction == 0.01
            assert schedule[i].n_cycles == 1

    def test_advanced_api_preserves_pruning_params(self):
        cycles = [
            CycleConfig(
                'greedy',
                n_cycles=3,
                batch_fraction=0.01,
                pruning_strategy='score',
                pruning_params={'pruning_fraction': 0.3}
            )
        ]

        schedule = parse_cycle_schedule(cycles=cycles)

        assert len(schedule) == 3
        for config in schedule:
            assert config.pruning_strategy == 'score'
            assert config.pruning_params == {'pruning_fraction': 0.3}

    def test_advanced_api_preserves_acquisition_params(self):
        cycles = [
            CycleConfig(
                'ucb',
                n_cycles=2,
                batch_fraction=0.01,
                acquisition_params={'exploration_weight': 2.0}
            )
        ]

        schedule = parse_cycle_schedule(cycles=cycles)

        assert len(schedule) == 2
        for config in schedule:
            assert config.acquisition_params == {'exploration_weight': 2.0}

    def test_invalid_cycles_not_list(self):
        with pytest.raises(ValueError, match="cycles must be a list"):
            parse_cycle_schedule(cycles="not_a_list")

    def test_invalid_cycles_wrong_type(self):
        with pytest.raises(ValueError, match="must be a CycleConfig instance"):
            parse_cycle_schedule(cycles=[{'strategy': 'greedy'}])

    def test_expansion_into_single_cycle_configs(self):
        cycles = [
            CycleConfig('random', n_cycles=5, batch_fraction=0.02)
        ]

        schedule = parse_cycle_schedule(cycles=cycles)

        assert len(schedule) == 5
        for config in schedule:
            assert config.n_cycles == 1
