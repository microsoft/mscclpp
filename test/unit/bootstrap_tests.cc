#include <gtest/gtest.h>
#include <mscclpp/core.hpp>
#include <cassert>
#include <iostream>
#include <memory>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

class BootstrapTest : public ::testing::Test {
protected:
  void SetUp() override {
    ipPort = "127.0.0.1:50000"; // IP and port to use for the bootstrap
  }

  void TearDown() override {
  }

  std::string ipPort;
};

TEST_F(BootstrapTest, SingleAllGather) {
  auto bootstrap = std::make_shared<mscclpp::Bootstrap>(0, 1);
  std::vector<int> tmp(bootstrap->getNranks(), 0);
  tmp[bootstrap->getRank()] = bootstrap->getRank() + 1;
  bootstrap->allGather(tmp.data(), sizeof(int));
  for (int i = 0; i < bootstrap->getNranks(); i++) {
    ASSERT_EQ(tmp[i], i + 1);
  }
}

TEST_F(BootstrapTest, SingleBarrier) {
  auto bootstrap = std::make_shared<mscclpp::Bootstrap>(0, 1);
  bootstrap->barrier();
}

TEST_F(BootstrapTest, AllGather) {
  int numProcesses = 16; // Number of processes to create
  std::vector<pid_t> processIDs;
  processIDs.resize(numProcesses);
  for (int i = 0; i < numProcesses; i++) {
    pid_t pid = fork();

    // TODO(chhwang): gtest env will be messed up if a child process exits with a non-zero status.
    if (pid == 0) {
      // Child process
      auto bootstrap = std::make_shared<mscclpp::Bootstrap>(i, numProcesses);
      bootstrap->initialize(ipPort);

      // AllGather test
      std::vector<int> tmp(bootstrap->getNranks(), 0);
      tmp[bootstrap->getRank()] = bootstrap->getRank() + 1;
      bootstrap->allGather(tmp.data(), sizeof(int));

      for (int j = 0; j < bootstrap->getNranks(); j++) {
        assert(tmp[j] == j + 1);
      }
      exit(0);
    } else if (pid > 0) {
      // Parent process
      processIDs[i] = pid;
    } else {
      FAIL() << "Fork failed.";
    }
  }
  // Waits for child processes to complete
  for (int i = 0; i < numProcesses; i++) {
    int status;
    waitpid(processIDs[i], &status, 0);
    EXPECT_TRUE(WIFEXITED(status));
    EXPECT_EQ(0, WEXITSTATUS(status));
  }
}

TEST_F(BootstrapTest, Barrier) {
  int numProcesses = 16; // Number of processes to create
  std::vector<pid_t> processIDs;
  processIDs.resize(numProcesses);
  for (int i = 0; i < numProcesses; i++) {
    pid_t pid = fork();

    // TODO(chhwang): gtest env will be messed up if a child process exits with a non-zero status.
    if (pid == 0) {
      // Child process
      auto bootstrap = std::make_shared<mscclpp::Bootstrap>(i, numProcesses);
      bootstrap->initialize(ipPort);
      bootstrap->barrier();
      exit(0);
    } else if (pid > 0) {
      // Parent process
      processIDs[i] = pid;
    } else {
      FAIL() << "Fork failed.";
    }
  }
  // Waits for child processes to complete
  for (int i = 0; i < numProcesses; i++) {
    int status;
    waitpid(processIDs[i], &status, 0);
    EXPECT_TRUE(WIFEXITED(status));
    EXPECT_EQ(0, WEXITSTATUS(status));
  }
}

TEST_F(BootstrapTest, SendRecv) {
  int msg1 = 3;
  int msg2 = 4;
  int msg3 = 5;

  // Sender
  pid_t pidSend = fork();
  if (pidSend == 0) {
    auto bootstrap = std::make_shared<mscclpp::Bootstrap>(0, 2);
    bootstrap->initialize(ipPort);

    bootstrap->send(&msg1, sizeof(int), 1, 0);
    bootstrap->send(&msg2, sizeof(int), 1, 1);
    bootstrap->send(&msg3, sizeof(int), 1, 2);

    exit(0);
  } else if (pidSend < 0) {
    FAIL() << "Fork failed.";
  }

  // Receiver
  pid_t pidRecv = fork();
  if (pidRecv == 0) {
    auto bootstrap = std::make_shared<mscclpp::Bootstrap>(1, 2);
    bootstrap->initialize(ipPort);

    int received1 = 0;
    int received2 = 0;
    int received3 = 0;
    bootstrap->recv(&received1, sizeof(int), 0, 0);
    bootstrap->recv(&received2, sizeof(int), 0, 1);
    bootstrap->recv(&received3, sizeof(int), 0, 2);

    ASSERT_EQ(received1, msg1);
    ASSERT_EQ(received2, msg2);
    ASSERT_EQ(received3, msg3);

    exit(0);
  } else if (pidRecv < 0) {
    FAIL() << "Fork failed.";
  }

  // Waits for child processes to complete
  int status;
  waitpid(pidSend, &status, 0);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(0, WEXITSTATUS(status));
  waitpid(pidRecv, &status, 0);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(0, WEXITSTATUS(status));
}
