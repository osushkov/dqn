
#include "common/Common.hpp"
#include "connectfour/GameRules.hpp"
#include "learning/ExperienceMemory.hpp"
#include <iostream>

int main(int argc, char **argv) {
  auto state = connectfour::GameRules::Instance()->InitialState();
  auto em = make_unique<learning::ExperienceMemory>(1000);

  cout << state << endl;
  cout << "hello world" << endl;
  return 0;
}
