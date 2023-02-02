#define MSCCLPP_MAJOR 0
#define MSCCLPP_MINOR 1

#define MSCCLPP_VERSION (MSCCLPP_MAJOR * 100 + MSCCLPP_MINOR)


#define MSCCLPP_UNIQUE_ID_BYTES 128
typedef struct { char internal[MSCCLPP_UNIQUE_ID_BYTES]; } mscclppUniqueId;

/* Error type */
typedef enum { mscclppSuccess                 =  0,
               mscclppUnhandledCudaError      =  1
            } mscclppResult_t;

mscclppResult_t  mscclppGetUniqueId(mscclppUniqueId* uniqueId);

//mscclppResult_t  mscclppCommInitRank(mscclppComm_t* comm, int nranks, mscclppUniqueId commId, int rank);
//mscclppResult_t  mscclppCommDestroy(mscclppComm_t comm);