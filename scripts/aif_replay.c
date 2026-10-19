/* Metadata-only NVMeVirt AiF replay. Does not evaluate matrix products. */
#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE
#include <endian.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <linux/nvme_ioctl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

struct chip_demand { uint64_t useful, pages; uint32_t rounds, cold; };
struct command { uint32_t magic, version, nchips, reserved; struct chip_demand chip[16]; };
struct chip_demand_v2 {
    uint64_t useful, pages;
    uint32_t rounds, cold;
    uint64_t input_bytes, output_bytes;
    uint32_t matrix_commands, reserved;
};
struct command_v2 {
    uint32_t magic, version, nchips, control_ns, host_ns, reserved[3];
    struct chip_demand_v2 chip[16];
};
_Static_assert(sizeof(struct command) == 400, "wire ABI size");
_Static_assert(sizeof(struct command_v2) == 800, "v2 wire ABI size");
static uint64_t now_ns(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ULL + t.tv_nsec;
}
int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr,"usage: %s /dev/nvme0n1 workload.bin\n",argv[0]); return 2; }
    FILE *input = fopen(argv[2],"rb");
    if (!input) { perror("workload"); return 1; }
    int fd = open(argv[1], O_RDWR);
    if (fd < 0) { perror("device"); return 1; }
    void *page = NULL;
    if (posix_memalign(&page,4096,4096)) return 1;
    uint64_t count=0, total_ns=0, min_ns=UINT64_MAX, max_ns=0;
    unsigned versions=0;
    int result=1;
    for (;;) {
        memset(page,0,4096);
        size_t size=fread(page,1,12,input);
        if (!size && feof(input)) break;
        if (size != 12) { fprintf(stderr,"truncated workload header or read error\n"); goto done; }
        const struct command *header=page;
        uint32_t version=le32toh(header->version);
        uint32_t nchips=le32toh(header->nchips);
        if (le32toh(header->magic)!=0x31464941U || (version!=1 && version!=2 && version!=3) ||
            !nchips || nchips>16) {
            fprintf(stderr,"invalid workload magic/version/chip count at command %"PRIu64"\n",count);
            goto done;
        }
        size_t payload_size=version==1 ? sizeof(struct command) : sizeof(struct command_v2);
        size=fread((char *)page+12,1,payload_size-12,input);
        if (size!=payload_size-12) { fprintf(stderr,"truncated workload command\n"); goto done; }
        struct nvme_passthru_cmd cmd = {
            .opcode=0xc1, .nsid=1, .addr=(uintptr_t)page,
            .data_len=payload_size, .cdw10=payload_size, .timeout_ms=10000,
        };
        uint64_t before=now_ns();
        int status=ioctl(fd,NVME_IOCTL_IO_CMD,&cmd);
        uint64_t elapsed=now_ns()-before;
        if (status) { fprintf(stderr,"command %"PRIu64" NVMe status=%d%s%s\n",count,status,
                             status<0?" errno=":"",status<0?strerror(errno):""); goto done; }
        count++; total_ns+=elapsed;
        versions|=1U<<(version-1);
        if(elapsed<min_ns) min_ns=elapsed;
        if(elapsed>max_ns) max_ns=elapsed;
    }
    printf("{\"commands\":%"PRIu64",\"ioctl_wall_ns\":%"PRIu64",\"min_ioctl_ns\":%"PRIu64",\"max_ioctl_ns\":%"PRIu64",\"mode\":\"%s\"}\n",
           count,total_ns,count?min_ns:0,max_ns,
           versions==4?"metadata_only_int8_v3":versions==2?"metadata_only_int8_v2":
           versions>1?"metadata_only_int8_mixed":"metadata_only_int8_v1");
    result=0;
done:
    free(page); close(fd); fclose(input); return result;
}
