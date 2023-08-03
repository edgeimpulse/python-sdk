<p align="center">
    <a href="https://www.edgeimpulse.com/"><img src="https://events.edgeimpulse.com/hs-fs/hubfs/Edge%20Impulse%20Full%20Logo_RGB.png?width=1817&name=Edge%20Impulse%20Full%20Logo_RGB.png?raw=true" alt="Edge Impulse logo"/></a>
</p>

# Edge Impulse SDK

The official Python SDK for Edge Impulse is designed to help machine learning practitioners build and deploy models for embedded hardware and edge AI applications.

- Profile your model to estimate RAM, ROM, and inference speed
- Convert your model to C++ to deploy on edge hardware
- Interact with Edge Impulse projects to collect data, train models, and deploy them to edge devices

[Sign up for a free account →](https://studio.edgeimpulse.com/signup)

## Getting Started

Install the Edge Impulse Python SDK:

```sh
pip install edgeimpulse
```

Estimate RAM, ROM, and inference speed for a variety of hardware platforms:

```python
import edgeimpulse as ei

# Change to an API key from your Edge Impulse project
ei.API_KEY = "your-api-key"

# Print inference estimates
result = ei.model.profile(model="path/to/model")
result.summary()
```

To learn about the full functionality, see the resources below.

## Resources

- [Overview and getting started](https://docs.edgeimpulse.com/docs/edge-impulse-python-sdk/overview)
- [Tutorial demonstrating how to profile and deploy a model](https://docs.edgeimpulse.com/docs/edge-impulse-python-sdk/01-python-sdk-with-tf-keras)
- [Reference guide](https://docs.edgeimpulse.com/reference/python-sdk/edgeimpulse)
