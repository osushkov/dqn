
#include "ExperienceMemory.hpp"
#include <cassert>

using namespace learning;

// How much memory to purge (as a ratio of max size) when the occupancy is saturated.
static constexpr float PURGE_RATIO = 0.2f;

ExperienceMemory::ExperienceMemory(unsigned maxSize)
    : pastExperiences(maxSize), head(0), tail(0), occupancy(0) {}

void ExperienceMemory::AddExperiences(const vector<ExperienceMoment> &moments) {
  boost::upgrade_lock<boost::shared_mutex> lock(mutex);
  boost::upgrade_to_unique_lock<boost::shared_mutex> uniqueLock(lock);

  for (unsigned i = 0; i < moments.size(); i++) {
    pastExperiences[tail] = moments[i];
    tail = (tail + 1) % pastExperiences.size();
    occupancy++;

    if (occupancy == pastExperiences.size()) {
      purgeOldMemories();
    }
  }
}

vector<ExperienceMoment> ExperienceMemory::Sample(unsigned numSamples) const {
  boost::shared_lock<boost::shared_mutex> lock(mutex);

  vector<ExperienceMoment> result;
  result.reserve(numSamples);

  for (unsigned i = 0; i < numSamples; i++) {
    result.push_back(pastExperiences[wrappedIndex(rand())]);
  }

  return result;
}

unsigned ExperienceMemory::wrappedIndex(unsigned i) const {
  assert(occupancy > 0 && pastExperiences.size() > 0);
  return ((i % occupancy) + head) % pastExperiences.size();
}

void ExperienceMemory::purgeOldMemories(void) {
  unsigned purgeAmount = PURGE_RATIO * occupancy;
  occupancy -= purgeAmount;
  head = (head + purgeAmount) % pastExperiences.size();
}
