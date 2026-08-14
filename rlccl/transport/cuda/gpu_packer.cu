#include "gpu_packer.cuh"

// The production packer is an inline device routine invoked directly by the
// persistent transport consumer.  This translation unit keeps the backend
// independently compilable without introducing a per-descriptor launch API.
