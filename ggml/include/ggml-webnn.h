#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_WEBNN_NAME "WebNN"

GGML_BACKEND_API ggml_backend_t ggml_backend_webnn_init(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_webnn_reg(void);

#ifdef  __cplusplus
}
#endif
