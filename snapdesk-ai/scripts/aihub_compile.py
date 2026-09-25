import argparse

import qai_hub as hub


def parse_shape(text):
    return tuple(int(x) for x in text.split(","))


def parse_inputs(names, shapes, dtypes):
    name_list = names.split(",")
    shape_list = shapes.split(";")
    dtype_list = (dtypes or ",".join(["float32"] * len(name_list))).split(",")
    if not (len(name_list) == len(shape_list) == len(dtype_list)):
        raise SystemExit("--input-names, --shapes and --dtypes must have the same number of items")
    return {n: (parse_shape(s), d) for n, s, d in zip(name_list, shape_list, dtype_list)}


def main():
    parser = argparse.ArgumentParser(description="Compile an ONNX model for a Snapdragon device with Qualcomm AI Hub.")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--model", help="Path to the .onnx file")
    parser.add_argument("--device", default="Snapdragon X Elite CRD")
    parser.add_argument("--input-names", default="input", help="Comma separated input names, e.g. input_ids,attention_mask")
    parser.add_argument("--shapes", default="1,3,224,224", help="Semicolon separated shapes matching --input-names, e.g. 1,256;1,256")
    parser.add_argument("--dtypes", default="", help="Comma separated dtypes matching --input-names (default float32), e.g. int64,int64")
    parser.add_argument("--output", default="models/vision/classifier.onnx")
    parser.add_argument("--profile", action="store_true", help="Also run a profile job on a hosted device")
    args = parser.parse_args()

    if args.list_devices:
        for device in hub.get_devices():
            print(device.name)
        return
    if not args.model:
        parser.error("--model is required")

    input_specs = parse_inputs(args.input_names, args.shapes, args.dtypes)
    device = hub.Device(args.device)
    model = hub.upload_model(args.model)
    job = hub.submit_compile_job(
        model=model,
        device=device,
        input_specs=input_specs,
        options="--target_runtime precompiled_qnn_onnx --truncate_64bit_io",
    )
    print(f"Compile job: {job.url}")
    job.wait()
    job.get_target_model().download(args.output)
    print(f"Saved compiled model to {args.output}")
    if args.profile:
        profile = hub.submit_profile_job(model=job.get_target_model(), device=device)
        print(f"Profile job (runs on a Qualcomm-hosted device, not your PC): {profile.url}")


if __name__ == "__main__":
    main()