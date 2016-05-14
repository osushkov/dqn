#pragma once

#include "../common/Common.hpp"
#include "ExperienceMoment.hpp"

#include <boost/thread/shared_mutex.hpp>

namespace learning {

class ExperienceMemory {
  mutable boost::shared_mutex mutex;

  vector<ExperienceMoment> pastExperiences;
  unsigned head;
  unsigned tail;
  unsigned occupancy;

public:
  ExperienceMemory(unsigned maxSize);
  ~ExperienceMemory() = default;

  ExperienceMemory(const ExperienceMemory &other) = delete;
  ExperienceMemory(ExperienceMemory &&other) = delete;
  ExperienceMemory &operator=(const ExperienceMemory &other) = delete;

  void AddExperiences(const vector<ExperienceMoment> &moments);
  vector<ExperienceMoment> Sample(unsigned numSamples) const;

private:
  unsigned wrappedIndex(unsigned i) const;
  void purgeOldMemories(void);
};
}
