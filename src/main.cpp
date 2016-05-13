
#include "common/Common.hpp"
#include "connectfour/GameRules.hpp"
#include <iostream>

int main(int argc, char **argv) {
  auto state = connectfour::GameRules::Instance()->InitialState();
  cout << "hello world" << endl;
  return 0;
}
