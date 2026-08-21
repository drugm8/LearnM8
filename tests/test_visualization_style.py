import matplotlib.pyplot as plt
import pytest

from learnm8.visualization import style

pytestmark = pytest.mark.unit


def test_compound_tick_label_shows_count_and_pool_fraction():
    assert style.compound_tick_label(250_000, 1_000_000) == '250k (25%)'
    assert style.compound_tick_label(1_000_000, 10_000_000) == '1.0M (10%)'


def test_style_axes_gives_labels_less_weight_than_titles():
    style.apply()
    fig, ax = plt.subplots()
    ax.set_title('Discovery')
    ax.set_xlabel('Compounds evaluated')
    ax.set_ylabel('Recovery (%)')

    style.style_axes(ax)

    assert ax.title.get_fontweight() == 'bold'
    assert ax.xaxis.label.get_fontweight() == 'semibold'
    assert ax.yaxis.label.get_fontweight() == 'semibold'
    plt.close(fig)


def test_compound_axis_replaces_ticks_with_count_and_fraction():
    style.apply()
    fig, ax = plt.subplots()
    ax.set_xticks([250_000, 500_000])

    style.set_compound_axis(ax, 1_000_000)

    assert [tick.get_text() for tick in ax.get_xticklabels()] == [
        '250k (25%)',
        '500k (50%)',
    ]
    assert {tick.get_rotation() for tick in ax.get_xticklabels()} == {45.0}
    assert ax.get_xlabel() == 'Evaluated compounds'
    plt.close(fig)


def test_compound_axis_uses_plotted_evaluation_points():
    style.apply()
    fig, ax = plt.subplots()
    ax.plot([10_000, 13_000, 19_000], [1, 2, 3])

    style.set_compound_axis(ax, 1_000_000)

    assert list(ax.get_xticks()) == [10_000, 13_000, 19_000]
    assert [tick.get_text() for tick in ax.get_xticklabels()] == [
        '10k (1%)',
        '13k (1.3%)',
        '19k (1.9%)',
    ]
    plt.close(fig)
