<p align="center">
    <a href="https://edgeimpulse.com/"><img src="https://cdn.edgeimpulse.com/images/edge-impulse-primary-logo-black-text-white-bg.png" alt="Edge Impulse logo"/></a>
</p>

# Edge Impulse SDK

The official Python SDK for Edge Impulse is designed to help machine learning practitioners build and deploy models for embedded hardware and edge AI applications.

- Profile your model to estimate RAM, ROM, and inference speed
- Convert your model to C++ to deploy on edge hardware
- Interact with Edge Impulse projects to collect data, train models, and deploy them to edge devices

[Sign up for a free account →](https://studio.edgeimpulse.com/signup)

List of versions and changes can be found [in this changelog](https://github.com/edgeimpulse/python-sdk/blob/main/CHANGELOG.md).

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
- [API reference guide](https://docs.edgeimpulse.com/reference/python-sdk/overview)
