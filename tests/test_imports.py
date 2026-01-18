"""Test that all main package imports work correctly."""

import pytest


class TestChasmImports:
    """Test CHASM package imports."""

    def test_chasm_base(self):
        """Test base chasm import."""
        import chasm

        assert chasm is not None

    def test_datasets(self):
        """Test dataset imports."""
        from chasm.datasets import aia_dataset, chasm_dataset, combined_dataset

    def test_gui(self):
        """Test GUI imports."""
        from chasm.gui.modules import gui_builder, image_manager

    def test_preprocessing(self):
        """Test preprocessing imports."""
        from chasm.preprocessing import load_fits


class TestChronnosImports:
    """Test CHRONNOS package imports."""

    def test_chronnos_base(self):
        """Test base chronnos import."""
        import chronnos

        assert chronnos is not None

    def test_chronnos_data(self):
        """Test chronnos data imports."""
        from chronnos.data.convert import sdo_cmaps
        from chronnos.data.generator import (
            CombinedCHDataset,
            MapDataset,
            MaskDataset,
        )

    def test_chronnos_train(self):
        """Test chronnos training imports."""
        from chronnos.train.model import CHRONNOS, Trainer
        from chronnos.train.callback import PlotCallback, ValidationCallback


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
