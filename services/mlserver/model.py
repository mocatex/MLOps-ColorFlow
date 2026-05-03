from mlserver import MLModel
from mlserver.types import InferenceRequest, InferenceResponse, ResponseOutput


class LinearRegressionModel(MLModel):
    async def load(self) -> bool:
        self.ready = True
        return self.ready

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        if not payload.inputs:
            raise ValueError("At least one input tensor is required")

        raw_values = payload.inputs[0].data or []
        x_values = [float(value) for value in raw_values]
        predictions = [(2.0 * value) + 1.0 for value in x_values]

        return InferenceResponse(
            model_name=self.name,
            outputs=[
                ResponseOutput(
                    name="predictions",
                    shape=[len(predictions)],
                    datatype="FP64",
                    data=predictions,
                )
            ],
        )