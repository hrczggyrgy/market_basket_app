"""Tests to ensure no hardcoded mock segment uplift data exists in UI code."""

import re
import pytest
from pathlib import Path


class TestNoMockSegmentUpliftData:
    """Ensure no fabricated segment labels or uplift values exist in UI source."""

    @pytest.fixture
    def ui_files(self):
        """Get all Python files in src/ui/."""
        ui_dir = Path(__file__).parent.parent / "src" / "ui"
        return list(ui_dir.glob("*.py"))

    def test_no_fabricated_segment_labels(self, ui_files):
        """Fail if the exact mock label list appears anywhere in UI code."""
        # This exact combination should never appear in production code
        forbidden_labels = ["High Value", "Regular", "Occasional", "New"]
        
        for file_path in ui_files:
            content = file_path.read_text()
            # Check if ALL four labels appear in the same file (co-occurrence)
            found_labels = [label for label in forbidden_labels if label in content]
            if len(found_labels) == 4:
                # They all appear - check if they're in a list/dict together
                # Use regex to detect the specific pattern of a segment list
                pattern = r'["\']High Value["\'].*["\']Regular["\'].*["\']Occasional["\'].*["\']New["\']'
                if re.search(pattern, content, re.DOTALL):
                    pytest.fail(
                        f"Fabricated segment labels found in {file_path.name}: "
                        f"'High Value', 'Regular', 'Occasional', 'New' co-occur "
                        f"in a data structure. This is the mock data pattern."
                    )

    def test_no_hardcoded_uplift_literals_in_render_function(self, ui_files):
        """Fail if _render_uplift_by_segment contains numeric uplift assignments."""
        for file_path in ui_files:
            if file_path.name == "pricing_tab.py":
                content = file_path.read_text()
                # Find the _render_uplift_by_segment function
                func_match = re.search(
                    r'def _render_uplift_by_segment\(.*?\):(.*?)(?:\ndef |\nclass |\Z)',
                    content,
                    re.DOTALL
                )
                if func_match:
                    func_body = func_match.group(1)
                    # Check for hardcoded uplift DataFrame creation with numeric literals
                    # Pattern: pd.DataFrame({... "uplift": [0.3, 0.15, ...] ...})
                    uplift_pattern = r'pd\.DataFrame\(\s*\{[^}]*"uplift"\s*:\s*\[[0-9.]'
                    if re.search(uplift_pattern, func_body):
                        pytest.fail(
                            f"Hardcoded numeric uplift values found in _render_uplift_by_segment "
                            f"in {file_path.name}. Real uplift must come from model output."
                        )
                    # Also check for the old mock assignment pattern
                    mock_pattern = r'segment_uplift\s*=\s*pd\.DataFrame'
                    if re.search(mock_pattern, func_body):
                        pytest.fail(
                            f"Direct DataFrame assignment to segment_uplift found in "
                            f"_render_uplift_by_segment in {file_path.name}. "
                            f"Must use real model output."
                        )

    def test_no_mock_return_in_train_function(self, ui_files):
        """Fail if _train_uplift_model returns a hardcoded dict with segment_uplift."""
        for file_path in ui_files:
            if file_path.name == "pricing_tab.py":
                content = file_path.read_text()
                func_match = re.search(
                    r'def _train_uplift_model\(.*?\):(.*?)(?:\ndef |\nclass |\Z)',
                    content,
                    re.DOTALL
                )
                if func_match:
                    func_body = func_match.group(1)
                    # Check for the old hardcoded return pattern
                    if '"segment_uplift"' in func_body and '"High Value"' in func_body:
                        pytest.fail(
                            f"_train_uplift_model still returns hardcoded mock segment_uplift "
                            f"in {file_path.name}"
                        )
                    # Check for any direct numeric literal assignment to qini/auuc
                    if re.search(r'"qini"\s*:\s*0\.\d+', func_body):
                        pytest.fail(
                            f"_train_uplift_model contains hardcoded qini value in {file_path.name}"
                        )
                    if re.search(r'"auuc"\s*:\s*0\.\d+', func_body):
                        pytest.fail(
                            f"_train_uplift_model contains hardcoded auuc value in {file_path.name}"
                        )

    def test_segment_uplift_uses_real_model_output(self, ui_files):
        """Verify _train_uplift_model calls real promo_uplift functions."""
        for file_path in ui_files:
            if file_path.name == "pricing_tab.py":
                content = file_path.read_text()
                # Check entire file for required function calls (function body is long)
                # Must call build_uplift_dataset
                assert "build_uplift_dataset" in content, (
                    "_train_uplift_model must call build_uplift_dataset"
                )
                # Must call train_t_learner_uplift or train_s_learner_uplift
                assert ("train_t_learner_uplift" in content 
                        or "train_s_learner_uplift" in content), (
                    "_train_uplift_model must call real uplift training functions"
                )
                # Must call evaluate_uplift_model
                assert "evaluate_uplift_model" in content, (
                    "_train_uplift_model must call evaluate_uplift_model"
                )
                # Must check session_state for segment_assignments
                assert 'session_state.get("segment_assignments")' in content, (
                    "_train_uplift_model must read segment_assignments from session_state"
                )


class TestSegmentUpliftEndToEnd:
    """Integration-style test: real pipeline produces real segment labels."""

    def test_real_segment_labels_match_segmentation_module(self):
        """
        This test documents the expected behavior:
        When segmentation runs, the segment labels in session_state['segment_assignments']
        must come from segmentation.py's output (rfm_segmentation, behavioral_segmentation, 
        or value_based_segmentation), not from a fixed list.
        
        Actual end-to-end test would require running Streamlit app which is complex.
        This test serves as documentation of the contract.
        """
        # The contract: segment_assignments["segment"].unique() 
        # must match one of the segmentation methods' output labels
        # Not a hardcoded list like ["High Value", "Regular", "Occasional", "New"]
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])