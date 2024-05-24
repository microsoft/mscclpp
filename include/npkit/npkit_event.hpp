// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef NPKIT_EVENT_H_
#define NPKIT_EVENT_H_

#define NPKIT_EVENT_INVALID                                 0x0

#define NPKIT_EVENT_TIME_SYNC_GPU                           0x1
#define NPKIT_EVENT_TIME_SYNC_CPU                           0x2

#define NPKIT_EVENT_EXECUTOR_BARRIER_ENTRY                  0x3
#define NPKIT_EVENT_EXECUTOR_BARRIER_EXIT                   0x4

#define NPKIT_EVENT_EXECUTOR_SIGNAL_ENTRY                   0x5
#define NPKIT_EVENT_EXECUTOR_SIGNAL_EXIT                    0x6

#define NPKIT_EVENT_EXECUTOR_WAIT_ENTRY                     0x7
#define NPKIT_EVENT_EXECUTOR_WAIT_EXIT                      0x8

#define NPKIT_EVENT_EXECUTOR_PUT_ENTRY                      0x9
#define NPKIT_EVENT_EXECUTOR_PUT_EXIT                       0x10

#define NPKIT_EVENT_EXECUTOR_GET_ENTRY                      0x11
#define NPKIT_EVENT_EXECUTOR_GET_EXIT                       0x12

#define NPKIT_EVENT_EXECUTOR_READ_REDUCE_COPY_SEND_ENTRY    0x13
#define NPKIT_EVENT_EXECUTOR_READ_REDUCE_COPY_SEND_EXIT     0x14

#define NPKIT_EVENT_EXECUTOR_READ_REDUCE_COPY_ENTRY         0x15
#define NPKIT_EVENT_EXECUTOR_READ_REDUCE_COPY_EXIT          0x1A

#define NPKIT_EVENT_EXECUTOR_PUT_PACKET_ENTRY               0x1B
#define NPKIT_EVENT_EXECUTOR_PUT_PACKET_EXIT                0x1C

#define NPKIT_EVENT_EXECUTOR_REDUCE_SEND_PACKET_ENTRY       0x1D
#define NPKIT_EVENT_EXECUTOR_REDUCE_SEND_PACKET_EXIT        0x1E

#define NPKIT_EVENT_EXECUTOR_COPY_PACKET_ENTRY              0x1F
#define NPKIT_EVENT_EXECUTOR_COPY_PACKET_EXIT               0x20

#define NPKIT_EVENT_EXECUTOR_REDUCE_SEND_ENTRY              0x21
#define NPKIT_EVENT_EXECUTOR_REDUCE_SEND_EXIT               0x22

#endif
