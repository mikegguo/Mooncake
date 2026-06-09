#pragma once

#include <string>
#include <exception>

#ifndef EP_STATIC_ASSERT
#define EP_STATIC_ASSERT(cond, reason) static_assert(cond, reason)
#endif

class EPException : public std::exception {
   private:
    std::string message = {};

   public:
    explicit EPException(const char* name, const char* file, const int line,
                         const std::string& error) {
        message = std::string("Failed: ") + name + " error " + file + ":" +
                  std::to_string(line) + " '" + error + "'";
    }

    const char* what() const noexcept override { return message.c_str(); }
};

// EP_CHECK is defined in mooncake_ep_device.h (unified CUDA/MUSA).
// CUDA_CHECK is kept as an alias for backward compatibility.
#ifndef CUDA_CHECK
#define CUDA_CHECK(cmd) EP_CHECK(cmd)
#endif

#ifndef EP_HOST_ASSERT
#define EP_HOST_ASSERT(cond)                                           \
    do {                                                               \
        if (not(cond)) {                                               \
            throw EPException("Assertion", __FILE__, __LINE__, #cond); \
        }                                                              \
    } while (0)
#endif

#ifndef EP_DEVICE_ASSERT
#define EP_DEVICE_ASSERT(cond)                                           \
    do {                                                                 \
        if (not(cond)) {                                                 \
            printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, \
                   __LINE__, #cond);                                     \
            __trap();                                                    \
        }                                                                \
    } while (0)
#endif
