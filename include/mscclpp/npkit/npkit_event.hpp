// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef NPKIT_EVENT_H_
#define NPKIT_EVENT_H_

#define NPKIT_EVENT_INVALID                                 0x0

#define NPKIT_EVENT_TIME_SYNC_GPU                           0x1
#define NPKIT_EVENT_TIME_SYNC_CPU                           0x2

#define NPKIT_EVENT_EXECUTOR_INIT_ENTRY                     0x3
#define NPKIT_EVENT_EXECUTOR_INIT_EXIT                      0x4

#define NPKIT_EVENT_EXECUTOR_OP_BASE_ENTRY                  0x5
#define NPKIT_EVENT_EXECUTOR_OP_BASE_EXIT                   0x15

#endif
