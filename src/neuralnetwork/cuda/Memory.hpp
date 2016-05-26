#pragma once

namespace neuralnetwork {
namespace cuda {
namespace memory {

void *AllocPushBuffer(size_t bufSize);
void FreePushBuffer(void *buf);
}
}
}
