cmake_minimum_required(VERSION 3.13)

if (NOT EMSCRIPTEN)
    message(FATAL_ERROR "GGML_WEBNN requires an Emscripten build: WebNN is only reachable "
                        "through the browser's navigator.ml API (use emcmake cmake ...)")
endif()

ggml_add_backend_library(ggml-webnn
    ggml-webnn.cpp
    ../../include/ggml-webnn.h
)

# WebNN calls (createContext, build, readTensor, ...) are async JS promises;
# EM_ASYNC_JS needs either JSPI or ASYNCIFY to suspend the wasm stack
if (GGML_WEBNN_JSPI)
    target_compile_options(ggml-webnn PRIVATE "-fwasm-exceptions")
    target_link_options(ggml-webnn INTERFACE "-sJSPI" "-fwasm-exceptions")
else()
    target_compile_options(ggml-webnn PRIVATE "-fexceptions")
    target_link_options(ggml-webnn INTERFACE "-sASYNCIFY" "-exceptions")
endif()

if (GGML_WEBNN_DEBUG)
    target_compile_definitions(ggml-webnn PRIVATE GGML_WEBNN_DEBUG=1)
    target_link_options(ggml-webnn INTERFACE "-sASSERTIONS=2")
endif()
