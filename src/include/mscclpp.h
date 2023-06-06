#ifndef MSCCLPP_H_
#define MSCCLPP_H_

// TODO: deprecate this file

#ifdef __cplusplus
extern "C" {
#endif

/* Error type */
typedef enum {
  mscclppSuccess = 0,
  mscclppUnhandledCudaError = 1,
  mscclppSystemError = 2,
  mscclppInternalError = 3,
  mscclppInvalidArgument = 4,
  mscclppInvalidUsage = 5,
  mscclppRemoteError = 6,
  mscclppInProgress = 7,
  mscclppNumResults = 8
} mscclppResult_t;

/* Return the string for the given error code.
 *
 * Output:
 *   returns the string
 *
 * Inputs:
 *   result: the error code that this function needs to translate
 */
const char* mscclppGetErrorString(mscclppResult_t result);

/* Log handler type which is a callback function for
 * however user likes to handle the log messages. Once set,
 * the logger will just call this function with msg.
 */
typedef void (*mscclppLogHandler_t)(const char* msg);

/* The default log handler.
 *
 * Inputs:
 *   msg: the log message
 */
void mscclppDefaultLogHandler(const char* msg);

/* Set a custom log handler.
 *
 * Inputs:
 *   handler: the log handler function
 */
mscclppResult_t mscclppSetLogHandler(mscclppLogHandler_t handler);

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // MSCCLPP_H_
