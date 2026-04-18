import argh
import pathlib
import logging
import logging.config
import typing as t
import yaml
import dataclasses
import enum

import torch
import torch.cuda
import torchaudio

import sam_audio
import sam_audio.model


@dataclasses.dataclass(frozen=True)
class ModelTypeData:

    name: str

    url: str


@enum.unique
class ModelType(enum.Enum):

    LARGE = ModelTypeData(
        name='facebook/sam-audio-large',
        url='https://huggingface.co/facebook/sam-audio-large',
    )

    SMALL = ModelTypeData(
        name='facebook/sam-audio-small',
        url='https://huggingface.co/facebook/sam-audio-small',
    )

    BASE = ModelTypeData(
        name='facebook/sam-audio-base',
        url='https://huggingface.co/facebook/sam-audio-base',
    )

    @property
    def data(self) -> ModelTypeData:
        return self.value

    @classmethod
    def parse(cls, value: str) -> t.Self:
        return cls[value]


@enum.unique
class DeviceType(enum.Enum):

    CUDA = 'cuda'

    CPU = 'cpu'

    AUTO = 'auto'

    @property
    def device(self) -> torch.device:
        match self:
            case DeviceType.CPU:
                return torch.device('cpu')
            case DeviceType.CUDA:
                return torch.device('cuda')
            case DeviceType.AUTO:
                return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            case _:
                raise ValueError('Unsupported device type: ' + self.name)

    @classmethod
    def parse(cls, value: str) -> t.Self:
        return cls[value]


class Application:
    """
    Facebook SAM model

    https://huggingface.co/facebook/sam-audio-base
    """

    PATH_APPLICATION: t.Final[pathlib.Path] = pathlib.Path(__file__)

    PATH_DIR_SOURCES: t.Final[pathlib.Path] = PATH_APPLICATION.parent.resolve()

    PATH_DIR_PACKAGE: t.Final[pathlib.Path] = PATH_DIR_SOURCES.parent.resolve()

    DTYPE: t.Final[torch.dtype] = torch.bfloat16

    def __init__(self):
        logging_config_path: pathlib.Path = self.PATH_DIR_SOURCES / 'sam.logging.yaml'
        logging_config = self.load_yaml(logging_config_path, yaml.SafeLoader)
        logging.config.dictConfig(logging_config)
        logging.info('using logging configuration [%s]', logging_config_path)

        self.logger = logging.getLogger('application')

    @argh.arg('--source', type=str, help='path to the audio file', required=True)
    @argh.arg('--query', type=str, help='query string', required=True)
    @argh.arg('--model-type', type=ModelType.parse, help='SAM model type (LARGE, SMALL, BASE)', required=False)
    @argh.arg('--device-type', type=DeviceType.parse, help='device type (CPU, CUDA, AUTO)', required=False)
    def main(
        self,
        source: str = None,
        query: str = None,
        model_type: ModelType = ModelType.SMALL,
        device_type: DeviceType = DeviceType.AUTO,
    ) -> int:
        self.logger.info('using source : %s', source)
        self.logger.info('using query  : %s', query)
        self.logger.info('using model  : %s', model_type.name)

        source_file_path: pathlib.Path = pathlib.Path(source)
        if not source_file_path.is_file():
            raise ValueError('Specified source file does not exists: ' + source)

        device: torch.device = device_type.device
        self.logger.info('using device : %s', device)

        processor: sam_audio.SAMAudioProcessor = sam_audio.SAMAudioProcessor.from_pretrained(
            model_name_or_path=model_type.data.name,
        )

        inputs: sam_audio.Batch = processor(
            audios=[source],
            descriptions=[query],
        )

        inputs = inputs.to(device=device)

        model: sam_audio.SAMAudio = sam_audio.SAMAudio.from_pretrained(
            pretrained_model_name_or_path=model_type.data.name,
        )

        model = model.to(device=device, dtype=self.DTYPE)
        model = model.eval()

        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=self.DTYPE):
                result: sam_audio.model.SeparationResult = model.separate(batch=inputs, predict_spans=True)

        torchaudio.save(
            uri=source_file_path.with_stem(f'{source_file_path.stem}-stem').with_suffix('.wav'),
            src=result.target[0].unsqueeze(0).cpu(),
            sample_rate=processor.audio_sampling_rate,
        )

        torchaudio.save(
            uri=source_file_path.with_stem(f'{source_file_path.stem}-residual').with_suffix('.wav'),
            src=result.residual[0].unsqueeze(0).cpu(),
            sample_rate=processor.audio_sampling_rate,
        )

        return 0

    @staticmethod
    def load_yaml(path: pathlib.Path, yaml_loader_class: t.Type) -> t.Dict:
        """
        Load YAML configuration from a file.

        Args:
            path: path to the YAML file
            yaml_loader_class: YAML loader class to use

        Returns:
            parsed YAML content as a dictionary
        """
        with path.open('rt') as file:
            yaml_text = file.read()

        # noinspection PyTypeChecker
        yaml_dict = yaml.load(yaml_text, yaml_loader_class)

        return yaml_dict


if __name__ == '__main__':
    application = Application()
    try:
        argh.dispatch_command(application.main)
    finally:
        logging.info('the work is finished')
        logging.shutdown()
