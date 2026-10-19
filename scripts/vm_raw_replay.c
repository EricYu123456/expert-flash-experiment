/* Real O_DIRECT reads through NVMeVirt's conventional FTL; no AiF compute. */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

struct request { uint64_t offset; size_t length; };
static struct request *requests;
static size_t nrequests, max_length, total_requests;
static atomic_size_t next_request;
static atomic_int failed;
static int device_fd;
struct worker_stats { uint64_t bytes, requests, latency_ns, max_latency_ns; };

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

static void *worker(void *arg)
{
    struct worker_stats *stats = arg;
    void *buffer = NULL;
    if (posix_memalign(&buffer, 4096, max_length)) {
        atomic_store(&failed, 1);
        return NULL;
    }
    while (!atomic_load(&failed)) {
        size_t index = atomic_fetch_add(&next_request, 1);
        if (index >= total_requests)
            break;
        struct request req = requests[index % nrequests];
        uint64_t start = now_ns();
        ssize_t size;
        do {
            size = pread(device_fd, buffer, req.length, (off_t)req.offset);
        } while (size < 0 && errno == EINTR);
        uint64_t elapsed = now_ns() - start;
        if (size < 0 || (size_t)size != req.length) {
            fprintf(stderr, "pread offset=%" PRIu64 " length=%zu returned %zd: %s\n",
                    req.offset, req.length, size, size < 0 ? strerror(errno) : "short read");
            atomic_store(&failed, 1);
            break;
        }
        stats->bytes += size;
        stats->requests++;
        stats->latency_ns += elapsed;
        if (elapsed > stats->max_latency_ns)
            stats->max_latency_ns = elapsed;
    }
    free(buffer);
    return NULL;
}

int main(int argc, char **argv)
{
    if (argc < 3 || argc > 5) {
        fprintf(stderr, "usage: %s DEVICE REQUESTS.csv [WORKERS=1] [REPETITIONS=1]\n"
                        "CSV rows: byte_offset,byte_length; aligned to 4096; length <= 262144\n", argv[0]);
        return 2;
    }
    unsigned workers = argc > 3 ? strtoul(argv[3], NULL, 10) : 1;
    unsigned repetitions = argc > 4 ? strtoul(argv[4], NULL, 10) : 1;
    if (!workers || workers > 64 || !repetitions || repetitions > 100000) {
        fprintf(stderr, "invalid workers/repetitions\n");
        return 2;
    }
    FILE *input = fopen(argv[2], "r");
    if (!input) { perror("CSV"); return 1; }
    char *line = NULL;
    size_t line_capacity = 0, capacity = 0;
    while (getline(&line, &line_capacity, input) >= 0) {
        uint64_t offset, length;
        char trailing;
        if (line[0] == '#' || !strncmp(line, "offset", 6))
            continue;
        if (sscanf(line, "%" SCNu64 ",%" SCNu64 " %c", &offset, &length, &trailing) != 2 ||
            !length || length > 262144 || offset % 4096 || length % 4096 ||
            offset > INT64_MAX - length) {
            fprintf(stderr, "invalid CSV row: %s", line);
            return 2;
        }
        if (nrequests == capacity) {
            capacity = capacity ? capacity * 2 : 1024;
            void *resized = realloc(requests, capacity * sizeof(*requests));
            if (!resized) { perror("realloc"); return 1; }
            requests = resized;
        }
        requests[nrequests++] = (struct request){offset, length};
        if (length > max_length) max_length = length;
    }
    free(line);
    fclose(input);
    if (!nrequests || nrequests > SIZE_MAX / repetitions) {
        fprintf(stderr, "empty or oversized workload\n");
        return 2;
    }
    total_requests = nrequests * repetitions;
    device_fd = open(argv[1], O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (device_fd < 0) { perror("device O_DIRECT"); return 1; }
    pthread_t *threads = calloc(workers, sizeof(*threads));
    struct worker_stats *stats = calloc(workers, sizeof(*stats));
    if (!threads || !stats) { perror("calloc"); return 1; }
    uint64_t start = now_ns();
    unsigned started = 0;
    for (; started < workers; started++) {
        int result = pthread_create(&threads[started], NULL, worker, &stats[started]);
        if (result) {
            fprintf(stderr, "pthread_create: %s\n", strerror(result));
            atomic_store(&failed, 1);
            break;
        }
    }
    struct worker_stats sum = {0};
    for (unsigned i = 0; i < started; i++) {
        pthread_join(threads[i], NULL);
        sum.bytes += stats[i].bytes;
        sum.requests += stats[i].requests;
        sum.latency_ns += stats[i].latency_ns;
        if (stats[i].max_latency_ns > sum.max_latency_ns) sum.max_latency_ns = stats[i].max_latency_ns;
    }
    uint64_t wall_ns = now_ns() - start;
    printf("{\"path\":\"conventional_nvmevirt_odirect\",\"workers\":%u,"
           "\"requests\":%" PRIu64 ",\"bytes\":%" PRIu64 ",\"wall_ns\":%" PRIu64 ","
           "\"mean_latency_ns\":%.1f,\"max_latency_ns\":%" PRIu64 ",\"failed\":%d}\n",
           workers, sum.requests, sum.bytes, wall_ns,
           sum.requests ? (double)sum.latency_ns / sum.requests : 0,
           sum.max_latency_ns, atomic_load(&failed));
    close(device_fd);
    free(requests); free(threads); free(stats);
    return atomic_load(&failed) ? 1 : 0;
}
