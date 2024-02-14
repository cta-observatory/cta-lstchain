import sys
from tqdm.auto import tqdm

from lstchain.image.image_processor import LSTImageProcessor
from lstchain.image.cleaning import LSTImageCleaner
from lstchain.image.muon.muon_processor import LSTMuonProcessor
from lstchain.calib.camera.interleaved_processor import LSTInterleavedProcessor
from lstchain.calib.camera.calibrator import LSTCameraCalibrator
from lstchain.mc.nsb_waveform_tuner import WaveformNSBTuner
from lstchain.reco.lhfit_processor import LHFitProcessor

from ctapipe.calib import GainSelector
from ctapipe.containers import EventType
from ctapipe.core import QualityQuery, Tool
from ctapipe.core.traits import Bool, classes_with_traits, flag
from ctapipe.image import ImageModifier
from ctapipe.image.extractor import ImageExtractor
from ctapipe.io import (
    DataLevel,
    DataWriter,
    EventSource,
    SimTelEventSource,
    metadata,
    write_table,
)
from ctapipe.io.datawriter import DATA_MODEL_VERSION
from ctapipe.reco import Reconstructor, ShowerProcessor

COMPATIBLE_DATALEVELS = [
    DataLevel.R1,
    DataLevel.DL0,
    DataLevel.DL1_IMAGES,
    DataLevel.DL1_PARAMETERS,
]

__all__ = ["LSTProcessorTool"]


class LSTProcessorTool(Tool):
    """
    Process data from lower-data levels up to DL1 and DL2, including image
    extraction and optionally image parameterization as well as muon analysis
    and shower reconstruction.

    Note that the muon analysis and shower reconstruction both depend on
    parametrized images and therefore compute image parameters even if
    DataWriter.write_parameters=False in case these are not already present
    in the input file.
    """

    name = "lstchain_process"
    description = (
        __doc__ + f" This currently uses data model version {DATA_MODEL_VERSION}"
    )
    examples = """
    To process data with all default values:
    > lstchain_process --input events.simtel.gz --output events.dl1.h5 --progress

    Or use an external configuration file, where you can specify all options:
    > lstchain_process -i events.simtel.gz --config lstchain_cat_A_config.json --progress

    For recalibrating or applying Cat-B calibrations on already processed DL1b data:
    > lstchain_process --input events_cat_A.dl1.h5 --cat-B-calibrations cat_B.h5
      --config lstchain_cat_A_config lstchain_cat_B_config --output events_cat_B.dl1.h5

    """

    progress_bar = Bool(
        help="show progress bar during processing", default_value=False
    ).tag(config=True)

    force_recompute_dl1 = Bool(
        help="Enforce dl1 recomputation even if already present in the input file",
        default_value=False,
    ).tag(config=True)

    force_recompute_dl2 = Bool(
        help="Enforce dl2 recomputation even if already present in the input file",
        default_value=False,
    ).tag(config=True)

    write_interleaved_only_mode = Bool(
        help="Only process and write interleaved events for calculating cat B "
        "calibration coefficients.",
        default_value=False,
    ).tag(config=True)

    aliases = {
        ("i", "input"): "EventSource.input_url",
        ("o", "output"): "DataWriter.output_path",
        ("b", "cat-B-calibrations"): "LSTCameraCalibrator.cat_b_calibrations_path",
        ("t", "allowed-tels"): "EventSource.allowed_tels",
        ("m", "max-events"): "EventSource.max_events",
        "reconstructor": "ShowerProcessor.reconstructor_types",
    }

    flags = {
        "overwrite": (
            {"DataWriter": {"overwrite": True}},
            "Overwrite output file if it exists",
        ),
        **flag(
            "progress",
            "ProcessorTool.progress_bar",
            "show a progress bar during event processing",
            "don't show a progress bar during event processing",
        ),
        **flag(
            "recompute-dl1",
            "ProcessorTool.force_recompute_dl1",
            "Enforce DL1 recomputation even if already present in the input file",
            "Only compute DL1 if there are no DL1b parameters in the file",
        ),
        **flag(
            "recompute-dl2",
            "ProcessorTool.force_recompute_dl2",
            "Enforce DL2 recomputation even if already present in the input file",
            "Only compute DL2 if there is no shower reconstruction in the file",
        ),
        **flag(
            "write-images",
            "DataWriter.write_images",
            "store DL1/Event/Telescope images in output",
            "don't store DL1/Event/Telescope images in output",
        ),
        **flag(
            "write-parameters",
            "DataWriter.write_parameters",
            "store DL1/Event/Telescope parameters in output",
            "don't store DL1/Event/Telescope parameters in output",
        ),
        **flag(
            "write-showers",
            "DataWriter.write_showers",
            "store DL2/Event parameters in output",
            "don't DL2/Event parameters in output",
        ),
        **flag(
            "write-index-tables",
            "DataWriter.write_index_tables",
            "generate PyTables index tables for the parameter and image datasets",
        ),
        **flag(
            "write-muon-parameters",
            "DataWriter.write_muon_parameters",
            "store DL1/Event/Telescope muon parameters in output",
            "don't store DL1/Event/Telescope muon parameters in output",
        ),
    }

    classes = (
        [
            LSTCameraCalibrator,
            DataWriter,
            LSTInterleavedProcessor,
            LSTImageProcessor,
            LSTMuonProcessor,
            LHFitProcessor,
            ShowerProcessor,
            metadata.Instrument,
            metadata.Contact,
        ]
        + classes_with_traits(EventSource)
        + classes_with_traits(LSTImageCleaner)
        + classes_with_traits(ImageExtractor)
        + classes_with_traits(GainSelector)
        + classes_with_traits(QualityQuery)
        + classes_with_traits(ImageModifier)
        + classes_with_traits(Reconstructor)
        + classes_with_traits(WaveformNSBTuner)
    )

    def setup(self):
        # setup components:
        self.event_source = self.enter_context(EventSource(parent=self))

        if not self.event_source.has_any_datalevel(COMPATIBLE_DATALEVELS):
            self.log.critical(
                "%s  needs the EventSource to provide at least one of these datalevels: %s"
                ", %s provides only %s",
                self.name,
                COMPATIBLE_DATALEVELS,
                self.event_source,
                self.event_source.datalevels,
            )
            sys.exit(1)

        subarray = self.event_source.subarray
        self.process_interleaved = LSTInterleavedProcessor(
            event_source=self.event_source,
            parent=self,
            subarray=subarray,
            write_only_mode=self.write_interleaved_only_mode,
        )
        self.tune_waveforms = WaveformNSBTuner(parent=self, subarray=subarray)
        self.calibrate = LSTCameraCalibrator(parent=self, subarray=subarray)
        self.process_images = LSTImageProcessor(parent=self, subarray=subarray)
        self.process_muons = LSTMuonProcessor(parent=self, subarray=subarray)
        self.process_lhfit = LHFitProcessor(parent=self, subarray=subarray)
        self.process_shower = ShowerProcessor(parent=self, subarray=subarray)
        self.write = self.enter_context(
            DataWriter(parent=self, event_source=self.event_source)
        )

        # warn if max_events prevents writing the histograms
        if (
            isinstance(self.event_source, SimTelEventSource)
            and self.event_source.max_events
            and self.event_source.max_events > 0
        ):
            self.log.warning(
                "No Simulated shower distributions will be written because "
                "EventSource.max_events is set to a non-zero number (and therefore "
                "shower distributions read from the input Simulation file are invalid)."
            )

    @property
    def should_compute_dl2(self):
        """returns true if we should compute DL2 info"""
        if self.write_interleaved_only_mode:
            return False

        if self.force_recompute_dl2:
            return True

        return self.write.write_showers

    @property
    def should_compute_dl1(self):
        """returns true if we should compute DL1 info"""
        if self.write_interleaved_only_mode:
            return False

        if self.force_recompute_dl1:
            return True

        if DataLevel.DL1_PARAMETERS in self.event_source.datalevels:
            return False

        return (
            self.write.write_parameters
            or self.should_compute_dl2
            or self.should_compute_muon_parameters
        )

    @property
    def should_calibrate(self):
        """returns true if data should be calibrated"""
        if self.write_interleaved_only_mode:
            return False

        if self.force_recompute_dl1:
            return True

        if (
            self.write.write_images
            and DataLevel.DL1_IMAGES not in self.event_source.datalevels
        ):
            return True

        if self.should_compute_dl1:
            return DataLevel.DL1_IMAGES not in self.event_source.datalevels

        return False

    @property
    def should_tune_waveforms(self):
        """returns true if we should add NSB in waveforms"""
        if self.write_interleaved_only_mode:
            return False

        if (
            self.event_source.is_simulation
            and self.tune_waveforms.apply_waveform_tuning
        ):
            return True

        return False

    @property
    def should_compute_muon_parameters(self):
        """returns true if we should compute muon parameters info"""
        if self.write_interleaved_only_mode:
            return False

        if self.write.write_muon_parameters:
            return True

        return False

    @property
    def should_compute_lhfit_parameters(self):
        """returns true if we should compute LHFit parameters"""
        if self.write_interleaved_only_mode:
            return False

        if self.process_lhfit.compute_lhfit_parameters:
            return True

        return False

    def _write_processing_statistics(self):
        """write out the event selection stats, etc."""
        # NOTE: don't remove this, not part of DataWriter
        if self.write_interleaved_only_mode:
            return False

        if self.should_compute_dl1:
            image_stats = self.process_images.check_image.to_table(functions=True)
            write_table(
                image_stats,
                self.write.output_path,
                path="/dl1/service/image_statistics",
                append=True,
            )

        if self.should_compute_dl2:
            reconstructors = self.process_shower.reconstructors
            reconstructor_names = self.process_shower.reconstructor_types
            for reconstructor_name, reconstructor in zip(
                reconstructor_names, reconstructors
            ):
                write_table(
                    reconstructor.quality_query.to_table(functions=True),
                    self.write.output_path,
                    f"/dl2/service/tel_event_statistics/{reconstructor_name}",
                    append=True,
                )

    def start(self):
        """
        Process events
        """
        self.log.info(
            "Only write interleaved events: %s", self.write_interleaved_only_mode
        )
        self.log.info("tuning NSB on waveforms: %s", self.tune_waveforms)
        self.log.info("applying calibration: %s", self.should_calibrate)
        self.log.info("(re)compute DL1: %s", self.should_compute_dl1)
        self.log.info("(re)compute DL2: %s", self.should_compute_dl2)
        self.log.info(
            "compute muon parameters: %s", self.should_compute_muon_parameters
        )
        self.log.info(
            "compute lhfit parameters: %s", self.should_compute_lhfit_parameters
        )
        self.event_source.subarray.info(printer=self.log.info)

        for event in tqdm(
            self.event_source,
            desc=self.event_source.__class__.__name__,
            total=self.event_source.max_events,
            unit="ev",
            disable=not self.progress_bar,
        ):
            self.log.debug("Processessing event_id=%s", event.index.event_id)

            if event.trigger.event_type in {
                EventType.FLATFIELD,
                EventType.SKY_PEDESTAL,
            }:
                self.process_interleaved(event)

            if self.should_tune_waveforms:
                self.tune_waveforms(event)

            if self.should_calibrate:
                self.calibrate(event)

            if self.should_compute_dl1:
                self.process_images(event)

            if self.should_compute_muon_parameters:
                self.process_muons(event)

            if self.should_compute_lhfit_parameters:
                self.process_lhfit(event)

            if self.should_compute_dl2:
                self.process_shower(event)

            if not self.write_interleaved_only_mode:
                self.write(event)

    def finish(self):
        """
        Last steps after processing events.
        """
        if not self.write_interleaved_only_mode:
            self.write.write_simulation_histograms(self.event_source)
            self._write_processing_statistics()


def main():
    """run the tool"""
    tool = LSTProcessorTool()
    tool.run()


if __name__ == "__main__":
    main()
