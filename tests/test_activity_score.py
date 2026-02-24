import pytest
from datetime import datetime, timedelta
import numpy as np
from pandas import DataFrame
from physiodsp.activity.activity_score import ActivityScore, ActivityScoreSettings


@pytest.mark.parametrize(
    "baseline_window_days",
    [15, 30]
)
def test_activity_score_excellent(baseline_window_days):
    """Test Activity Score for excellent performance (>=85)"""
    np.random.seed(42)
    settings = ActivityScoreSettings(baseline_window_days=baseline_window_days)
    algorithm = ActivityScore(settings=settings)

    # Create excellent baseline data with random variation
    dates = [datetime.now() - timedelta(days=x) for x in range(baseline_window_days, 0, -1)]
    baseline_data = {
        'date': dates,
        'steps': np.random.randint(12000, 13000, baseline_window_days).tolist(),
        'sleep_hours': np.random.uniform(8.0, 8.3, baseline_window_days).tolist(),
        'training_minutes': np.random.randint(65, 75, baseline_window_days).tolist(),
        'resting_minutes': np.random.randint(540, 570, baseline_window_days).tolist()
    }

    # Excellent current day (at or above baseline)
    current_day = {
        'date': datetime.now(),
        'steps': np.random.randint(12500, 13500),
        'sleep_hours': np.random.uniform(8.1, 8.4),
        'training_minutes': np.random.randint(70, 85),
        'resting_minutes': np.random.randint(550, 580)
    }

    all_data = DataFrame({
        **{k: v for k, v in baseline_data.items()},
        'date': baseline_data['date'] + [current_day['date']],
        'steps': baseline_data['steps'] + [current_day['steps']],
        'sleep_hours': baseline_data['sleep_hours'] + [current_day['sleep_hours']],
        'training_minutes': baseline_data['training_minutes'] + [current_day['training_minutes']],
        'resting_minutes': baseline_data['resting_minutes'] + [current_day['resting_minutes']]
    })

    result = algorithm.run(all_data)
    score = result.biomarker_agg.iloc[0]['activity_score']

    assert score >= 85, f"Expected excellent score (>=85), got {score}"
    assert algorithm.get_activity_score_interpretation(score) == "Excellent - Outstanding activity and recovery balance"


@pytest.mark.parametrize(
    "baseline_window_days",
    [15, 30]
)
def test_activity_score_good(baseline_window_days):
    """Test Activity Score for good performance (70-84)"""
    np.random.seed(43)
    settings = ActivityScoreSettings(baseline_window_days=baseline_window_days)
    algorithm = ActivityScore(settings=settings)

    # Create good baseline data with random variation
    dates = [datetime.now() - timedelta(days=x) for x in range(baseline_window_days, 0, -1)]
    baseline_data = {
        'date': dates,
        'steps': np.random.randint(8500, 9500, baseline_window_days).tolist(),
        'sleep_hours': np.random.uniform(7.2, 7.8, baseline_window_days).tolist(),
        'training_minutes': np.random.randint(40, 50, baseline_window_days).tolist(),
        'resting_minutes': np.random.randint(480, 520, baseline_window_days).tolist()
    }

    # Good current day
    current_day = {
        'date': datetime.now(),
        'steps': np.random.randint(9500, 10500),
        'sleep_hours': np.random.uniform(7.8, 8.2),
        'training_minutes': np.random.randint(48, 60),
        'resting_minutes': np.random.randint(510, 540)
    }

    all_data = DataFrame({
        **{k: v for k, v in baseline_data.items()},
        'date': baseline_data['date'] + [current_day['date']],
        'steps': baseline_data['steps'] + [current_day['steps']],
        'sleep_hours': baseline_data['sleep_hours'] + [current_day['sleep_hours']],
        'training_minutes': baseline_data['training_minutes'] + [current_day['training_minutes']],
        'resting_minutes': baseline_data['resting_minutes'] + [current_day['resting_minutes']]
    })

    result = algorithm.run(all_data)
    score = result.biomarker_agg.iloc[0]['activity_score']

    assert 70 <= score < 85, f"Expected good score (70-84), got {score}"
    assert algorithm.get_activity_score_interpretation(score) == "Good - Healthy activity levels with adequate recovery"


@pytest.mark.parametrize(
    "baseline_window_days",
    [15, 30]
)
def test_activity_score_fair(baseline_window_days):
    """Test Activity Score for fair performance (50-69)"""
    np.random.seed(44)
    settings = ActivityScoreSettings(baseline_window_days=baseline_window_days)
    algorithm = ActivityScore(settings=settings)

    # Create fair baseline data with random variation
    dates = [datetime.now() - timedelta(days=x) for x in range(baseline_window_days, 0, -1)]
    baseline_data = {
        'date': dates,
        'steps': np.random.randint(7500, 8500, baseline_window_days).tolist(),
        'sleep_hours': np.random.uniform(7.5, 8.0, baseline_window_days).tolist(),
        'training_minutes': np.random.randint(45, 55, baseline_window_days).tolist(),
        'resting_minutes': np.random.randint(500, 540, baseline_window_days).tolist()
    }

    # Fair current day (slightly below baseline)
    current_day = {
        'date': datetime.now(),
        'steps': np.random.randint(5500, 6500),
        'sleep_hours': np.random.uniform(6.8, 7.2),
        'training_minutes': np.random.randint(35, 45),
        'resting_minutes': np.random.randint(470, 500)
    }

    all_data = DataFrame({
        **{k: v for k, v in baseline_data.items()},
        'date': baseline_data['date'] + [current_day['date']],
        'steps': baseline_data['steps'] + [current_day['steps']],
        'sleep_hours': baseline_data['sleep_hours'] + [current_day['sleep_hours']],
        'training_minutes': baseline_data['training_minutes'] + [current_day['training_minutes']],
        'resting_minutes': baseline_data['resting_minutes'] + [current_day['resting_minutes']]
    })

    result = algorithm.run(all_data)
    score = result.biomarker_agg.iloc[0]['activity_score']

    assert 50 <= score < 70, f"Expected fair score (50-69), got {score}"
    assert algorithm.get_activity_score_interpretation(score) == "Fair - Room for improvement in activity or recovery"


@pytest.mark.parametrize(
    "baseline_window_days",
    [15, 30]
)
def test_activity_score_poor(baseline_window_days):
    """Test Activity Score for poor performance (30-49)"""
    np.random.seed(45)
    settings = ActivityScoreSettings(baseline_window_days=baseline_window_days)
    algorithm = ActivityScore(settings=settings)

    # Create poor baseline data with random variation
    dates = [datetime.now() - timedelta(days=x) for x in range(baseline_window_days, 0, -1)]
    baseline_data = {
        'date': dates,
        'steps': np.random.randint(5000, 6000, baseline_window_days).tolist(),
        'sleep_hours': np.random.uniform(5.5, 6.0, baseline_window_days).tolist(),
        'training_minutes': np.random.randint(15, 25, baseline_window_days).tolist(),
        'resting_minutes': np.random.randint(420, 460, baseline_window_days).tolist()
    }

    # Poor current day (significantly below baseline)
    current_day = {
        'date': datetime.now(),
        'steps': np.random.randint(2500, 3500),
        'sleep_hours': np.random.uniform(4.0, 4.8),
        'training_minutes': np.random.randint(5, 15),
        'resting_minutes': np.random.randint(300, 350)
    }

    all_data = DataFrame({
        **{k: v for k, v in baseline_data.items()},
        'date': baseline_data['date'] + [current_day['date']],
        'steps': baseline_data['steps'] + [current_day['steps']],
        'sleep_hours': baseline_data['sleep_hours'] + [current_day['sleep_hours']],
        'training_minutes': baseline_data['training_minutes'] + [current_day['training_minutes']],
        'resting_minutes': baseline_data['resting_minutes'] + [current_day['resting_minutes']]
    })

    result = algorithm.run(all_data)
    score = result.biomarker_agg.iloc[0]['activity_score']

    assert 30 <= score < 50, f"Expected poor score (30-49), got {score}"
    assert algorithm.get_activity_score_interpretation(score) == "Poor - Significant imbalance in activity or recovery"


@pytest.mark.parametrize(
    "baseline_window_days",
    [15, 30]
)
def test_activity_score_critical(baseline_window_days):
    """Test Activity Score for critical performance (<30)"""
    np.random.seed(46)
    settings = ActivityScoreSettings(baseline_window_days=baseline_window_days)
    algorithm = ActivityScore(settings=settings)

    # Create critical baseline data with random variation
    dates = [datetime.now() - timedelta(days=x) for x in range(baseline_window_days, 0, -1)]
    baseline_data = {
        'date': dates,
        'steps': np.random.randint(2500, 3500, baseline_window_days).tolist(),
        'sleep_hours': np.random.uniform(5.0, 5.5, baseline_window_days).tolist(),
        'training_minutes': np.random.randint(5, 15, baseline_window_days).tolist(),
        'resting_minutes': np.random.randint(380, 420, baseline_window_days).tolist()
    }

    # Critical current day (significantly below baseline)
    current_day = {
        'date': datetime.now(),
        'steps': np.random.randint(200, 500),
        'sleep_hours': np.random.uniform(2.5, 3.5),
        'training_minutes': np.random.randint(0, 2),
        'resting_minutes': np.random.randint(150, 200)
    }

    all_data = DataFrame({
        **{k: v for k, v in baseline_data.items()},
        'date': baseline_data['date'] + [current_day['date']],
        'steps': baseline_data['steps'] + [current_day['steps']],
        'sleep_hours': baseline_data['sleep_hours'] + [current_day['sleep_hours']],
        'training_minutes': baseline_data['training_minutes'] + [current_day['training_minutes']],
        'resting_minutes': baseline_data['resting_minutes'] + [current_day['resting_minutes']]
    })

    result = algorithm.run(all_data)
    score = result.biomarker_agg.iloc[0]['activity_score']

    assert score < 30, f"Expected critical score (<30), got {score}"
    assert algorithm.get_activity_score_interpretation(score) == "Critical - Urgent attention needed to activity and recovery"


def test_activity_score_algorithm_properties():
    """Test basic properties of Activity Score algorithm"""
    algorithm = ActivityScore()

    assert algorithm.algorithm_name == "ActivityScore"
    assert algorithm.version == "v0.1.0"

    default_settings = ActivityScoreSettings()
    assert default_settings.baseline_window_days == 30
    assert default_settings.step_weight == 0.25
    assert default_settings.sleep_weight == 0.35
    assert default_settings.training_weight == 0.25
    assert default_settings.resting_weight == 0.15


def test_activity_score_minimum_data():
    """Test that algorithm requires minimum 2 days of data"""
    settings = ActivityScoreSettings()
    algorithm = ActivityScore(settings=settings)

    # Only 1 day of data
    single_day = DataFrame({
        'steps': [8000],
        'sleep_hours': [8.0],
        'training_minutes': [60],
        'resting_minutes': [540]
    })

    with pytest.raises(ValueError, match="Need at least 2 days of data"):
        algorithm.run(single_day)


def test_activity_score_personalizes_to_baseline():
    """Test that scores are personalized based on baseline data"""
    np.random.seed(47)
    settings = ActivityScoreSettings(baseline_window_days=15)
    algorithm_low = ActivityScore(settings=settings)
    algorithm_high = ActivityScore(settings=settings)

    # Low baseline user with random variation
    dates = [datetime.now() - timedelta(days=x) for x in range(15, 0, -1)]
    low_baseline = DataFrame({
        'date': dates + [datetime.now()],
        'steps': np.random.randint(4800, 5200, 15).tolist() + [5500],
        'sleep_hours': np.random.uniform(6.3, 6.7, 15).tolist() + [7.0],
        'training_minutes': np.random.randint(18, 22, 15).tolist() + [25],
        'resting_minutes': np.random.randint(410, 430, 15).tolist() + [450]
    })

    # High baseline user with random variation and same current day
    high_baseline = DataFrame({
        'date': dates + [datetime.now()],
        'steps': np.random.randint(14500, 15500, 15).tolist() + [5500],
        'sleep_hours': np.random.uniform(8.3, 8.7, 15).tolist() + [7.0],
        'training_minutes': np.random.randint(115, 125, 15).tolist() + [25],
        'resting_minutes': np.random.randint(590, 610, 15).tolist() + [450]
    })

    result_low = algorithm_low.run(low_baseline)
    result_high = algorithm_high.run(high_baseline)

    score_low = result_low.biomarker_agg.iloc[0]['activity_score']
    score_high = result_high.biomarker_agg.iloc[0]['activity_score']

    # Same current day should yield different scores based on baseline
    assert score_low != score_high
    # Low baseline user with same data should score higher (meeting their baseline)
    assert score_low > score_high


def test_activity_score_output_structure():
    """Test output DataFrame structure"""
    settings = ActivityScoreSettings(baseline_window_days=15)
    algorithm = ActivityScore(settings=settings)

    dates = [datetime.now() - timedelta(days=x) for x in range(15, 0, -1)]
    data = DataFrame({
        'date': dates + [datetime.now()],
        'steps': [8000] * 16,
        'sleep_hours': [8.0] * 16,
        'training_minutes': [60] * 16,
        'resting_minutes': [540] * 16
    })

    result = algorithm.run(data)

    # Check biomarker structure
    assert hasattr(result, 'biomarker_agg')
    assert len(result.biomarker_agg) == 1

    # Check required columns
    required_cols = ['activity_score', 'step_score', 'sleep_score',
                     'training_score', 'resting_score', 'baseline_days_used']
    for col in required_cols:
        assert col in result.biomarker_agg.columns

    # Check score ranges
    assert 0 <= result.biomarker_agg.iloc[0]['activity_score'] <= 100
    assert 0 <= result.biomarker_agg.iloc[0]['step_score'] <= 100
    assert 0 <= result.biomarker_agg.iloc[0]['sleep_score'] <= 100
    assert 0 <= result.biomarker_agg.iloc[0]['training_score'] <= 100
    assert 0 <= result.biomarker_agg.iloc[0]['resting_score'] <= 100


def test_activity_score_weighted_combination():
    """Test that final score is correct weighted combination"""
    settings = ActivityScoreSettings(
        step_weight=0.25,
        sleep_weight=0.35,
        training_weight=0.25,
        resting_weight=0.15
    )
    algorithm = ActivityScore(settings=settings)

    dates = [datetime.now() - timedelta(days=x) for x in range(15, 0, -1)]
    data = DataFrame({
        'date': dates + [datetime.now()],
        'steps': [8000] * 16,
        'sleep_hours': [8.0] * 16,
        'training_minutes': [60] * 16,
        'resting_minutes': [540] * 16
    })

    result = algorithm.run(data)
    score = result.biomarker_agg.iloc[0]

    # Manually calculate expected score
    expected_activity_score = (
        score['step_score'] * 0.25 +
        score['sleep_score'] * 0.35 +
        score['training_score'] * 0.25 +
        score['resting_score'] * 0.15
    )

    assert np.isclose(score['activity_score'], expected_activity_score, atol=0.5)


def test_activity_score_baseline_stats_stored():
    """Test that baseline statistics are computed and stored"""
    settings = ActivityScoreSettings(baseline_window_days=15)
    algorithm = ActivityScore(settings=settings)

    dates = [datetime.now() - timedelta(days=x) for x in range(15, 0, -1)]
    data = DataFrame({
        'date': dates + [datetime.now()],
        'steps': list(range(5000, 5015)) + [6000],
        'sleep_hours': [7.0 + i*0.1 for i in range(15)] + [8.0],
        'training_minutes': [30 + i*2 for i in range(15)] + [60],
        'resting_minutes': [450 + i*5 for i in range(15)] + [500]
    })

    result = algorithm.run(data)

    assert result.baseline_stats is not None
    assert 'steps_median' in result.baseline_stats
    assert 'steps_std' in result.baseline_stats
    assert 'sleep_median' in result.baseline_stats
    assert 'training_median' in result.baseline_stats
    assert 'resting_median' in result.baseline_stats

    # Check that baseline stats make sense
    assert result.baseline_stats['steps_median'] > 0
    assert result.baseline_stats['sleep_median'] > 0
