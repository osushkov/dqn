
#pragma once

namespace learning {

static constexpr unsigned MOMENTS_BATCH_SIZE = 32;
static constexpr unsigned TARGET_FUNCTION_UPDATE_RATE = 1000;
static constexpr float REWARD_DELAY_DISCOUNT = 0.99f;
static constexpr float INITIAL_MAXQ_TEMPERATURE = 5.0f;
}
