import asyncio
import os
import tempfile

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    dummy_file_path: str

    def initialize(self, args):
        with tempfile.NamedTemporaryFile(delete=False) as outfile:
            size = 268435456 # 256mb
            outfile.write(b'0' * size)
            self.dummy_file_path = outfile.name

    async def execute(self, requests):
        awaits = []
        for r in requests:
            pb_utils.Logger.log_info(
                f'bls execute: {r.request_id()}'
            )
            with open(self.dummy_file_path, 'rb') as f:
                data = np.frombuffer(f.read(), dtype='uint8')
                data = data.reshape([1, len(data)])

                infer_request = pb_utils.InferenceRequest(
                    request_id=r.request_id(),
                    inputs=[
                        pb_utils.Tensor('data', data)
                    ],
                    requested_output_names=['data'],
                    model_name='dummy',
                    model_version=1,
                )

                awaits.append(infer_request.async_exec())

        responses = await asyncio.gather(*awaits)
        for r, response in zip(requests, responses):
            if response.has_error():
                raise pb_utils.TritonModelException(
                    response.error().message()
                )

            response = pb_utils.get_output_tensor_by_name(response, 'data')
            if response is not None:
                response = response.as_numpy()
                pb_utils.Logger.log_info(
                    f'Dummy request: {data.shape} - {response.shape}'
                )

            sender = r.get_response_sender()
            sender.send(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.get_input_tensor_by_name(r, 'dummy')
                    ]
                ),
                flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
            )

    def finalize(self):
        os.remove(self.dummy_file_path)
