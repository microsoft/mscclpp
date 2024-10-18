// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_IBVERBS_WRAPPER_HPP_
#define MSCCLPP_IBVERBS_WRAPPER_HPP_

#include <dlfcn.h>
#include <infiniband/verbs.h>

#include <iostream>
#include <mscclpp/errors.hpp>

namespace mscclpp {

struct IBVerbs {
 private:
  // Static method to initialize the library
  static void initialize() {
    initialized = true;
    handle = dlopen("libibverbs.so", RTLD_NOW);
    if (!handle) {
      throw mscclpp::IbError("Failed to load libibverbs: " + std::string(dlerror()), errno);
    }

    // Load the necessary functions
    ibv_get_device_list_lib = (ibv_get_device_list_t)dlsym(handle, "ibv_get_device_list");
    ibv_free_device_list_lib = (ibv_free_device_list_t)dlsym(handle, "ibv_free_device_list");
    ibv_alloc_pd_lib = (ibv_alloc_pd_t)dlsym(handle, "ibv_alloc_pd");
    ibv_dealloc_pd_lib = (ibv_dealloc_pd_t)dlsym(handle, "ibv_dealloc_pd");
    ibv_open_device_lib = (ibv_open_device_t)dlsym(handle, "ibv_open_device");
    ibv_close_device_lib = (ibv_close_device_t)dlsym(handle, "ibv_close_device");
    ibv_query_device_lib = (ibv_query_device_t)dlsym(handle, "ibv_query_device");
    ibv_create_cq_lib = (ibv_create_cq_t)dlsym(handle, "ibv_create_cq");
    ibv_create_qp_lib = (ibv_create_qp_t)dlsym(handle, "ibv_create_qp");
    ibv_destroy_cq_lib = (ibv_destroy_cq_t)dlsym(handle, "ibv_destroy_cq");
    ibv_reg_mr_lib = (ibv_reg_mr_t)dlsym(handle, "ibv_reg_mr");
    ibv_dereg_mr_lib = (ibv_dereg_mr_t)dlsym(handle, "ibv_dereg_mr");
    ibv_query_gid_lib = (ibv_query_gid_t)dlsym(handle, "ibv_query_gid");
    ibv_modify_qp_lib = (ibv_modify_qp_t)dlsym(handle, "ibv_modify_qp");
    ibv_destroy_qp_lib = (ibv_destroy_qp_t)dlsym(handle, "ibv_destroy_qp");
    ibv_query_port_lib = (ibv_query_port_t)dlsym(handle, "ibv_query_port");
    ibv_reg_mr_iova2_lib = (ibv_reg_mr_iova2_t)dlsym(handle, "ibv_reg_mr_iova2");

    if (!ibv_get_device_list_lib || !ibv_free_device_list_lib || !ibv_alloc_pd_lib || !ibv_dealloc_pd_lib ||
        !ibv_open_device_lib || !ibv_close_device_lib || !ibv_query_device_lib || !ibv_create_cq_lib ||
        !ibv_create_qp_lib || !ibv_destroy_cq_lib || !ibv_reg_mr_lib || !ibv_dereg_mr_lib || !ibv_query_gid_lib ||
        !ibv_reg_mr_iova2_lib || !ibv_modify_qp_lib || !ibv_destroy_qp_lib || !ibv_query_port_lib) {
      throw mscclpp::IbError("Failed to load one or more function in the ibibverbs library: " + std::string(dlerror()),
                             errno);
      dlclose(handle);
    }
  }

 public:
  // Static method to get the device list
  static struct ibv_device** ibv_get_device_list(int* num_devices) {
    if (!initialized) initialize();
    if (ibv_get_device_list_lib) {
      return ibv_get_device_list_lib(num_devices);
    }
    return nullptr;
  }

  // Static method to free the device list
  static void ibv_free_device_list(struct ibv_device** list) {
    if (!initialized) initialize();
    if (ibv_free_device_list_lib) {
      ibv_free_device_list_lib(list);
    }
  }

  // Static method to allocate a protection domain
  static struct ibv_pd* ibv_alloc_pd(struct ibv_context* context) {
    if (!initialized) initialize();
    if (ibv_alloc_pd_lib) {
      return ibv_alloc_pd_lib(context);
    }
    return nullptr;
  }

  // Static method to deallocate a protection domain
  static int ibv_dealloc_pd(struct ibv_pd* pd) {
    if (!initialized) initialize();
    if (ibv_dealloc_pd_lib) {
      return ibv_dealloc_pd_lib(pd);
    }
    return -1;
  }

  // Static method to open a device
  static struct ibv_context* ibv_open_device(struct ibv_device* device) {
    if (!initialized) initialize();
    if (ibv_open_device_lib) {
      return ibv_open_device_lib(device);
    }
    return nullptr;
  }

  // Static method to close a device
  static int ibv_close_device(struct ibv_context* context) {
    if (!initialized) initialize();
    if (ibv_close_device_lib) {
      return ibv_close_device_lib(context);
    }
    return -1;
  }

  // Static method to query a device
  static int ibv_query_device(struct ibv_context* context, struct ibv_device_attr* device_attr) {
    if (!initialized) initialize();
    if (ibv_query_device_lib) {
      return ibv_query_device_lib(context, device_attr);
    }
    return -1;
  }

  // Static method to create a completion queue
  static struct ibv_cq* ibv_create_cq(struct ibv_context* context, int cqe, void* cq_context,
                                      struct ibv_comp_channel* channel, int comp_vector) {
    if (!initialized) initialize();
    if (ibv_create_cq_lib) {
      return ibv_create_cq_lib(context, cqe, cq_context, channel, comp_vector);
    }
    return nullptr;
  }

  // Static method to create a queue pair
  static struct ibv_qp* ibv_create_qp(struct ibv_pd* pd, struct ibv_qp_init_attr* qp_init_attr) {
    if (!initialized) initialize();
    if (ibv_create_qp_lib) {
      return ibv_create_qp_lib(pd, qp_init_attr);
    }
    return nullptr;
  }

  // Static method to destroy a completion queue
  static int ibv_destroy_cq(struct ibv_cq* cq) {
    if (!initialized) initialize();
    if (ibv_destroy_cq_lib) {
      return ibv_destroy_cq_lib(cq);
    }
    return -1;
  }

  // Static method to register a memory region
  static struct ibv_mr* ibv_reg_mr2(struct ibv_pd* pd, void* addr, size_t length, int access) {
    if (!initialized) initialize();
    if (ibv_reg_mr_lib) {
      return ibv_reg_mr_lib(pd, addr, length, access);
    }
    return nullptr;
  }

  // Static method to deregister a memory region
  static int ibv_dereg_mr(struct ibv_mr* mr) {
    if (!initialized) initialize();
    if (ibv_dereg_mr_lib) {
      return ibv_dereg_mr_lib(mr);
    }
    return -1;
  }

  // Static method to query a GID
  static int ibv_query_gid(struct ibv_context* context, uint8_t port_num, int index, union ibv_gid* gid) {
    if (!initialized) initialize();
    if (ibv_query_gid_lib) {
      return ibv_query_gid_lib(context, port_num, index, gid);
    }
    return -1;
  }

  // Static method to modify a queue pair
  static int ibv_modify_qp(struct ibv_qp* qp, struct ibv_qp_attr* attr, int attr_mask) {
    if (!initialized) initialize();
    if (ibv_modify_qp_lib) {
      return ibv_modify_qp_lib(qp, attr, attr_mask);
    }
    return -1;
  }

  // Static method to destroy a queue pair
  static int ibv_destroy_qp(struct ibv_qp* qp) {
    if (!initialized) initialize();
    if (ibv_destroy_qp_lib) {
      return ibv_destroy_qp_lib(qp);
    }
    return -1;
  }

  static inline int ibv_post_send(struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** bad_wr) {
    if (!initialized) initialize();
    return qp->context->ops.post_send(qp, wr, bad_wr);
  }

  static inline int ibv_poll_cq(struct ibv_cq* cq, int num_entries, struct ibv_wc* wc) {
    if (!initialized) initialize();
    return cq->context->ops.poll_cq(cq, num_entries, wc);
  }

  static int ibv_query_port_w(struct ibv_context* context, uint8_t port_num, struct ibv_port_attr* port_attr) {
    if (!initialized) initialize();
    if (ibv_query_port_lib) {
      return ibv_query_port_lib(context, port_num, port_attr);
    }
    return -1;
  }

  static struct ibv_mr* ibv_reg_mr_iova2_w(struct ibv_pd* pd, void* addr, size_t length, uint64_t iova,
                                           unsigned int access) {
    if (!initialized) initialize();
    if (ibv_reg_mr_iova2_lib) {
      return ibv_reg_mr_iova2_lib(pd, addr, length, iova, access);
    }
    return nullptr;
  }

  // Static method to clean up
  static void cleanup() {
    if (handle) {
      dlclose(handle);
      handle = nullptr;
    }
  }

 private:
  // Handle for the dynamic library
  static inline void* handle = nullptr;

  // Function pointers
  typedef struct ibv_device** (*ibv_get_device_list_t)(int*);
  typedef void (*ibv_free_device_list_t)(struct ibv_device**);
  typedef struct ibv_pd* (*ibv_alloc_pd_t)(struct ibv_context*);
  typedef int (*ibv_dealloc_pd_t)(struct ibv_pd*);
  typedef struct ibv_context* (*ibv_open_device_t)(struct ibv_device*);
  typedef int (*ibv_close_device_t)(struct ibv_context*);
  typedef int (*ibv_query_device_t)(struct ibv_context*, struct ibv_device_attr*);
  typedef struct ibv_cq* (*ibv_create_cq_t)(struct ibv_context*, int, void*, struct ibv_comp_channel*, int);
  typedef struct ibv_qp* (*ibv_create_qp_t)(struct ibv_pd*, struct ibv_qp_init_attr*);
  typedef int (*ibv_destroy_cq_t)(struct ibv_cq*);
  typedef int (*ibv_destroy_qp_t)(struct ibv_qp*);
  typedef struct ibv_mr* (*ibv_reg_mr_t)(struct ibv_pd*, void*, size_t, int);
  typedef int (*ibv_dereg_mr_t)(struct ibv_mr*);
  typedef int (*ibv_query_gid_t)(struct ibv_context*, uint8_t, int, union ibv_gid*);
  typedef int (*ibv_modify_qp_t)(struct ibv_qp*, struct ibv_qp_attr*, int);
  typedef int (*ibv_query_port_t)(struct ibv_context*, uint8_t, struct ibv_port_attr*);
  typedef struct ibv_mr* (*ibv_reg_mr_iova2_t)(struct ibv_pd* pd, void* addr, size_t length, uint64_t iova,
                                               unsigned int access);

  static inline ibv_get_device_list_t ibv_get_device_list_lib;
  static inline ibv_free_device_list_t ibv_free_device_list_lib = nullptr;
  static inline ibv_alloc_pd_t ibv_alloc_pd_lib = nullptr;
  static inline ibv_dealloc_pd_t ibv_dealloc_pd_lib = nullptr;
  static inline ibv_open_device_t ibv_open_device_lib = nullptr;
  static inline ibv_close_device_t ibv_close_device_lib = nullptr;
  static inline ibv_query_device_t ibv_query_device_lib = nullptr;
  static inline ibv_create_cq_t ibv_create_cq_lib = nullptr;
  static inline ibv_create_qp_t ibv_create_qp_lib = nullptr;
  static inline ibv_destroy_cq_t ibv_destroy_cq_lib = nullptr;
  static inline ibv_reg_mr_t ibv_reg_mr_lib = nullptr;
  static inline ibv_dereg_mr_t ibv_dereg_mr_lib = nullptr;
  static inline ibv_query_gid_t ibv_query_gid_lib = nullptr;
  static inline ibv_modify_qp_t ibv_modify_qp_lib = nullptr;
  static inline ibv_destroy_qp_t ibv_destroy_qp_lib = nullptr;
  static inline ibv_query_port_t ibv_query_port_lib = nullptr;
  static inline ibv_reg_mr_iova2_t ibv_reg_mr_iova2_lib = nullptr;

  static inline bool initialized = false;
};

}  // namespace mscclpp

#endif  // MSCCLPP_IBVERBS_WRAPPER_HPP_