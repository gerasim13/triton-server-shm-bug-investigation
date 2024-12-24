import asyncio

import triton_python_backend_utils as pb_utils


def cause_error():
    raise pb_utils.TritonModelException('Dummy error')


class TritonPythonModel:
    async def dummy(self, r):
        pb_utils.Logger.log_info(
            f'dummy execute: {r.request_id()}'
        )
        return pb_utils.InferenceResponse(
            output_tensors=[
                pb_utils.get_input_tensor_by_name(r, 'data')
            ]
        )

    async def execute(self, requests):
        awaits = []
        for r in requests:
            awaits.append(self.dummy(r))

        cause_error()

        return await asyncio.gather(*awaits)
