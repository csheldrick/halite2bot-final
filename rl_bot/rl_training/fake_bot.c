#include "fake_bot.h"

#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>

int closeFlag = 0;

void term(int signum) {
    closeFlag = 1;
}

int main(int argc, char * argv[]) {

    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_handler = term;
    sigaction(SIGTERM, &action, NULL);
    
    if (argc != 2) {
        fprintf(stdout, "fake_bot did not get fifo ID!");
        exit(EXIT_FAILURE);
    }

    int fifoID = atoi(argv[1]);

    char toFifoName[strlen(TO_HALITE_PREFIX) + 2];
    char fromFifoName[strlen(FROM_HALITE_PREFIX) + 2];

    snprintf(toFifoName, strlen(TO_HALITE_PREFIX) + 2, "%s%d", TO_HALITE_PREFIX, fifoID);
    snprintf(fromFifoName, strlen(FROM_HALITE_PREFIX) + 2, "%s%d", FROM_HALITE_PREFIX, fifoID);
    mkfifo(toFifoName,0666);
    mkfifo(fromFifoName, 0666);
    // open named pipes
    int fromPipeFd = open(fromFifoName, O_WRONLY | O_TRUNC);
    int toPipeFd = open(toFifoName, O_RDONLY | O_TRUNC);

    if (fromPipeFd < 0) {
        perror("from pipe open");
        exit(EXIT_FAILURE);
    }

    if (toPipeFd < 0) {
        perror("to pipe open");
        exit(EXIT_FAILURE);
    }

    // STDIN -> from_pipe fd
    // to_pipe fd -> STDOUT

    int n;
    char buf[BUFSIZ];
    //char logName[16];
    //int log;
    switch (fork()) {
        case -1:
            perror("Fork failed.");
            exit(EXIT_FAILURE);
        case 0: // Child

            //snprintf(logName, 11, "child%d.log", fifoID);
            //log = open(logName, O_WRONLY | O_CREAT | O_TRUNC, 0666);

            close(toPipeFd);

            while ((n = read(STDIN_FILENO, buf, BUFSIZ)) > 0) {

                if (closeFlag) {
                    //write(log, "earera.\n", 7);
                    //printf("here");
                    close(fromPipeFd);
                    exit(EXIT_SUCCESS);
                }

                //int other = write(log, buf, n);
                int bytesWritten = write(fromPipeFd, buf, n);
            }

            

            int bytes = write(fromPipeFd, "Done.\n", 6);
            //write(log, "Done.\n", 6);

            close(fromPipeFd);
            //close(log);
            break;
        default: // Parent

            //snprintf(logName, 12, "parent%d.log", fifoID);
            //log = open(logName, O_WRONLY | O_CREAT | O_TRUNC, 0666);

            close(fromPipeFd);
            while ((n = read(toPipeFd, buf, BUFSIZ)) > 0) {
                //int other = write(log, buff, n);
                if (closeFlag) {
                    //write(log, "earera.\n", 7);
                    //printf("here");
                    close(toPipeFd);
                    exit(EXIT_SUCCESS);
                }

                int bytesWritten = write(STDOUT_FILENO, buf, n);
            }

            close(toPipeFd);
    }
    return EXIT_SUCCESS;

}

