import numpy as np
import torch
from torch import nn
import torchlibrosa


class WaveToSpectrogram(nn.Module):

    def __init__( self, n_fft, hop_length, center=False ):
        super(WaveToSpectrogram, self).__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length

        self.spec_extractor = torchlibrosa.stft.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            center=center,
        )


        self.input_name = 'input.1'
        self.output_name = '14' # looked up via Netron
        
    def forward( self, x ):
        return self.spec_extractor( x )

    def gen_torch_output( self, sample_input ):
        self.eval()
        with torch.no_grad():
            torch_output = self( torch.from_numpy( sample_input ) )
            torch_output = torch_output.cpu().detach().numpy()
        return torch_output

    def convert_to_onnx( self, filename_onnx, sample_input ):

        input_names = [ self.input_name ]
        output_names = [ self.output_name ]
        
        torch.onnx.export(
            self,
            torch.from_numpy( sample_input ),
            filename_onnx,
            input_names=input_names,
            output_names=output_names,
            # operator_export_type=OperatorExportTypes.ONNX
        )
        
    def convert_to_coreml( self, fn_mlmodel, sample_input ):
        import onnx
        import onnx_coreml
        
        torch_spectrogram = self.gen_torch_output( sample_input )
        print( 'torch_spectrogram: shape %s\nsample %s ' % ( torch_spectrogram.shape, torch_spectrogram[:, :, :3, :3] ) )

        # first convert to ONNX
        filename_onnx = '/tmp/wave__spectrogram_model.onnx'
        model.convert_to_onnx( filename_onnx, sample_input )


        # set up for Core ML export
        convert_params = dict(
            predicted_feature_name = [],
            minimum_ios_deployment_target='13',
        )

        mlmodel = onnx_coreml.convert(
            model=filename_onnx,
            **convert_params, 
        )

        '''
        output = spec.description.output[0]
        import coremltools.proto.FeatureTypes_pb2 as ft
        output.type.imageType.colorSpace = ft.ImageFeatureType.GRAYSCALE
        output.type.imageType.height = 300
        output.type.imageType.width = 150
        '''

        assert mlmodel != None, 'CoreML Conversion failed'

        mlmodel.save( fn_mlmodel )

        model_inputs = dict(
            self.input_name = sample_input
        )
        # do forward pass
        mlmodel_outputs = mlmodel.predict(model_inputs, useCPUOnly=True)

        # fetch the spectrogram from output dictionary
        mlmodel_spectrogram =  mlmodel_outputs[ self.output_name ]
        print( 'mlmodel_output: shape %s sample %s ' % ( mlmodel_spectrogram.shape, mlmodel_spectrogram[:3, :3] ) )

        assert torch_spectrogram.shape == mlmodel_spectrogram.shape
        
        assert np.allclose( torch_spectrogram, mlmodel_spectrogram )
            
        print( 'Successful MLModel conversion to %s!' % fn_mlmodel )

        return mlmodel_output

def load_wav_file( fn_wav ):
    import soundfile as sf

    data, samplerate = sf.read( fn_wav )
    return data
    
def save_ml_model_output_as_json( fn_output, mlmodel_output ):
    import json
    with open( fn_output, 'w' ) as fp:
        json.dump( mlmodel_output.tolist(), fp )
    
if __name__ == '__main__':
    import sys
    fn_sample_wav = sys.argv[1]
    fn_mlmodel = sys.argv[2]
    fn_model_output = sys.argv[3]

    num_samples = 32000
    waveform = load_wav_file( fn_sample_wav )
    sample_input = waveform[ :num_samples ].astype( dtype=np.float32 )
    # shape: (samples_num,)
    
    # add batch dimension
    sample_input = np.expand_dims( sample_input, axis=0 )
    # shape: (batch_size, samples_num)
    
    model = WaveToSpectrogram( n_fft=1024, hop_length=512 )

    mlmodel_output = model.convert_to_coreml( fn_mlmodel, sample_input )
    # shape: (1, 1, 61, 513)

    save_ml_model_output_as_json( fn_model_output, mlmodel_output[0,0,...])

'''
# example command:
python model.py ../../sample_assets/bonjour.wav /tmp/wave__spec.mlmodel  /tmp/spec_out.bonjour.json
'''    
