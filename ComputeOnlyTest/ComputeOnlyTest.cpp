#define NOMINMAX
#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>

#include <stdio.h>
#include <algorithm>

class ThreadingOptions
{
public:
	OrtThreadingOptions* threadingOptions = nullptr;

	ThreadingOptions()
	{
		const OrtApi& ortApi = Ort::GetApi();

		ortApi.CreateThreadingOptions(&threadingOptions);

		ortApi.SetGlobalIntraOpNumThreads(threadingOptions, 0);
		ortApi.SetGlobalInterOpNumThreads(threadingOptions, 0);
	}
	~ThreadingOptions()
	{
		const OrtApi& ortApi = Ort::GetApi();

		ortApi.ReleaseThreadingOptions(threadingOptions);
	}
};


static void ORT_API_CALL LoggingFunction(void* /*param*/, OrtLoggingLevel severity, const char* /*category*/, const char* /*logid*/, const char* /*code_location*/, const char* message)
{
	printf("onnxruntime: %s\n", message);
}

static Ort::Env& getOrtEnv()
{
	static Ort::Env s_OrtEnv{ nullptr };

	if (s_OrtEnv == nullptr)
	{
		ThreadingOptions threadingOptions;

		s_OrtEnv = Ort::Env(threadingOptions.threadingOptions, LoggingFunction, nullptr, ORT_LOGGING_LEVEL_VERBOSE, "");

		s_OrtEnv.DisableTelemetryEvents();
	}

	return s_OrtEnv;
}

int main()
{
	const OrtApi& ortApi = Ort::GetApi();

	const OrtDmlApi* ortDmlApi = nullptr;

	auto sessionOptions = Ort::SessionOptions{};

	int* device_ids = nullptr;

	Ort::ThrowOnError(ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ortDmlApi)));

	Ort::ThrowOnError(ortDmlApi->GetComputeOnlyDevices(device_ids));

	if (device_ids == nullptr)
	{
		printf("No compute only devices\n");
		return 0;
	}

	// Using the first compute device
	Ort::ThrowOnError(ortDmlApi->SessionOptionsAppendExecutionProvider_DML2(sessionOptions, *device_ids));

	sessionOptions.DisableMemPattern();
	sessionOptions.DisablePerSessionThreads();
	sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
	sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

	auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	auto session = Ort::Session(getOrtEnv(), L"..\\Dummy_model.onnx", sessionOptions);

	std::vector<int64_t> dims_input{ 1, 256, 128, 256, 1 };
	std::vector<int64_t> dims_output{ 1, 256, 128, 256, 1 };

	std::vector<Ort::Float16_t> vecInput(256 * 128 * 256);
	for (int i = 0; i < 256 * 128 * 256; i++)
	{
		vecInput[i] = 0;
	}

	Ort::Value inputTensor = Ort::Value::CreateTensor<Ort::Float16_t>(memoryInfo, vecInput.data(), vecInput.size(), dims_input.data(), dims_input.size());

	const char* inputName = "input";
	const char* outputName = "conv3d_18";

	auto outputValues = session.Run(Ort::RunOptions{ nullptr }, &inputName, &inputTensor, 1, &outputName, 1);

	const Ort::Float16_t* pOutputData = outputValues[0].GetTensorData<Ort::Float16_t>();

	uint16_t minOutput = 65535;
	uint16_t maxOutput = 0;
	for (int i = 0; i < 256 * 128 * 256; i++)
	{
		uint16_t r = pOutputData[i];
		minOutput = std::min(minOutput, r);
		maxOutput = std::max(maxOutput, r);
	}

	printf("min/max: %d/%d\n", minOutput, maxOutput);

}


