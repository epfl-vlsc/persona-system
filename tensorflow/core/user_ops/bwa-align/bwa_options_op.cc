
// Stuart Byma
// Op providing SNAP genome index and genome

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <memory>
#include <utility>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/user_ops/object-pool/basic_container.h"
#include "bwa/bwamem.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;

    class BWAOptionsOp : public OpKernel {
    public:
      typedef BasicContainer<mem_opt_t> BWAOptionsContainer;

        BWAOptionsOp(OpKernelConstruction* context)
            : OpKernel(context), options_handle_set_(false) {
          
          OP_REQUIRES_OK(context, context->GetAttr("options", &options_));
          
          OP_REQUIRES_OK(context,
                         context->allocate_persistent(DT_STRING, TensorShape({ 2 }),
                                                      &options_handle_, nullptr));
        }

        void Compute(OpKernelContext* ctx) override {
            mutex_lock l(mu_);
            if (!options_handle_set_) {
                OP_REQUIRES_OK(ctx, SetOptionsHandle(ctx, options_));
            }
            ctx->set_output_ref(0, &mu_, options_handle_.AccessTensor(ctx));
        }

    protected:
        ~BWAOptionsOp() override {
            // If the genome object was not shared, delete it.
            if (options_handle_set_ && cinfo_.resource_is_private_to_kernel()) {
                TF_CHECK_OK(cinfo_.resource_manager()->Delete<BWAOptionsContainer>(
                    cinfo_.container(), cinfo_.name()));
            }
        }

    protected:

        ContainerInfo cinfo_;

    private:
        void update_a(mem_opt_t *opt, const mem_opt_t *opt0)
        {
          if (opt0->a) { // matching score is changed
            if (!opt0->b) opt->b *= opt->a;
            if (!opt0->T) opt->T *= opt->a;
            if (!opt0->o_del) opt->o_del *= opt->a;
            if (!opt0->e_del) opt->e_del *= opt->a;
            if (!opt0->o_ins) opt->o_ins *= opt->a;
            if (!opt0->e_ins) opt->e_ins *= opt->a;
            if (!opt0->zdrop) opt->zdrop *= opt->a;
            if (!opt0->pen_clip5) opt->pen_clip5 *= opt->a;
            if (!opt0->pen_clip3) opt->pen_clip3 *= opt->a;
            if (!opt0->pen_unpaired) opt->pen_unpaired *= opt->a;
          }
        }

        Status SetOptionsHandle(OpKernelContext* ctx, vector<string> options_string) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            TF_RETURN_IF_ERROR(cinfo_.Init(ctx->resource_manager(), def()));
            BWAOptionsContainer* bwa_options;

            auto creator = [this, options_string](BWAOptionsContainer** options) {
                mem_opt_t* opt = mem_opt_init();
                mem_opt_t opt0;
                int fd, fd2, i, ignore_alt = 0, no_mt_io = 0;
                int fixed_chunk_size = -1;
                gzFile fp, fp2 = 0;
                char *p, *rg_line = 0, *hdr_line = 0;
                string mode = "";
                void *ko = 0, *ko2 = 0;
                mem_pestat_t pes[4];
                //ktp_aux_t aux;
                //process opts
                memset(&opt0, 0, sizeof(mem_opt_t));
                char c;
                const char* optarg;
                for (size_t i = 0; i < options_string.size(); i++) {
                  c = options_string[i][0]; // should just be one char ...
                  if (i != options_string.size() - 1)
                    optarg = options_string[i+1].c_str();
                  LOG(INFO) << "option is: " << options_string[i];

                  if (c == 'k') opt->min_seed_len = atoi(optarg), opt0.min_seed_len = 1;
                  else if (c == '1') no_mt_io = 1;
                  else if (c == 'x') mode = string(optarg);
                  else if (c == 'w') opt->w = atoi(optarg), opt0.w = 1;
                  else if (c == 'A') opt->a = atoi(optarg), opt0.a = 1;
                  else if (c == 'B') opt->b = atoi(optarg), opt0.b = 1;
                  else if (c == 'T') opt->T = atoi(optarg), opt0.T = 1;
                  else if (c == 'U') opt->pen_unpaired = atoi(optarg), opt0.pen_unpaired = 1;
                  else if (c == 't') opt->n_threads = atoi(optarg), opt->n_threads = opt->n_threads > 1? opt->n_threads : 1;
                  else if (c == 'P') opt->flag |= MEM_F_NOPAIRING;
                  else if (c == 'a') opt->flag |= MEM_F_ALL;
                  else if (c == 'p') opt->flag |= MEM_F_PE | MEM_F_SMARTPE;
                  else if (c == 'M') opt->flag |= MEM_F_NO_MULTI;
                  else if (c == 'S') opt->flag |= MEM_F_NO_RESCUE;
                  else if (c == 'Y') opt->flag |= MEM_F_SOFTCLIP;
                  else if (c == 'V') opt->flag |= MEM_F_REF_HDR;
                  else if (c == '5') opt->flag |= MEM_F_PRIMARY5;
                  else if (c == 'c') opt->max_occ = atoi(optarg), opt0.max_occ = 1;
                  else if (c == 'd') opt->zdrop = atoi(optarg), opt0.zdrop = 1;
                  else if (c == 'v') bwa_verbose = atoi(optarg);
                  else if (c == 'j') ignore_alt = 1;
                  else if (c == 'r') opt->split_factor = atof(optarg), opt0.split_factor = 1.;
                  else if (c == 'D') opt->drop_ratio = atof(optarg), opt0.drop_ratio = 1.;
                  else if (c == 'm') opt->max_matesw = atoi(optarg), opt0.max_matesw = 1;
                  else if (c == 's') opt->split_width = atoi(optarg), opt0.split_width = 1;
                  else if (c == 'G') opt->max_chain_gap = atoi(optarg), opt0.max_chain_gap = 1;
                  else if (c == 'N') opt->max_chain_extend = atoi(optarg), opt0.max_chain_extend = 1;
                  else if (c == 'W') opt->min_chain_weight = atoi(optarg), opt0.min_chain_weight = 1;
                  else if (c == 'y') opt->max_mem_intv = atol(optarg), opt0.max_mem_intv = 1;
                  else if (c == 'K') fixed_chunk_size = atoi(optarg);
                  else if (c == 'X') opt->mask_level = atof(optarg);
                  else if (c == 'h') {
                    opt0.max_XA_hits = opt0.max_XA_hits_alt = 1;
                    opt->max_XA_hits = opt->max_XA_hits_alt = strtol(optarg, &p, 10);
                    if (*p != 0 && ispunct(*p) && isdigit(p[1]))
                      opt->max_XA_hits_alt = strtol(p+1, &p, 10);
                  }
                  else if (c == 'Q') {
                    opt0.mapQ_coef_len = 1;
                    opt->mapQ_coef_len = atoi(optarg);
                    opt->mapQ_coef_fac = opt->mapQ_coef_len > 0? log(opt->mapQ_coef_len) : 0;
                  } else if (c == 'O') {
                    opt0.o_del = opt0.o_ins = 1;
                    opt->o_del = opt->o_ins = strtol(optarg, &p, 10);
                    if (*p != 0 && ispunct(*p) && isdigit(p[1]))
                      opt->o_ins = strtol(p+1, &p, 10);
                  } else if (c == 'E') {
                    opt0.e_del = opt0.e_ins = 1;
                    opt->e_del = opt->e_ins = strtol(optarg, &p, 10);
                    if (*p != 0 && ispunct(*p) && isdigit(p[1]))
                      opt->e_ins = strtol(p+1, &p, 10);
                  } else if (c == 'L') {
                    opt0.pen_clip5 = opt0.pen_clip3 = 1;
                    opt->pen_clip5 = opt->pen_clip3 = strtol(optarg, &p, 10);
                    if (*p != 0 && ispunct(*p) && isdigit(p[1]))
                      opt->pen_clip3 = strtol(p+1, &p, 10);
                  } else if (c == 'H') {
                    if (optarg[0] != '@') {
                      FILE *fp;
                      if ((fp = fopen(optarg, "r")) != 0) {
                        char *buf;
                        buf = (char*)calloc(1, 0x10000);
                        while (fgets(buf, 0xffff, fp)) {
                          i = strlen(buf);
                          assert(buf[i-1] == '\n'); // a long line
                          buf[i-1] = 0;
                          hdr_line = bwa_insert_header(buf, hdr_line);
                        }
                        free(buf);
                        fclose(fp);
                      }
                    } else hdr_line = bwa_insert_header(optarg, hdr_line);
                  } else if (c == 'I') { // specify the insert size distribution
                    //aux.pes0 = pes;
                    pes[1].failed = 0;
                    pes[1].avg = strtod(optarg, &p);
                    pes[1].std = pes[1].avg * .1;
                    if (*p != 0 && ispunct(*p) && isdigit(p[1]))
                      pes[1].std = strtod(p+1, &p);
                    pes[1].high = (int)(pes[1].avg + 4. * pes[1].std + .499);
                    pes[1].low  = (int)(pes[1].avg - 4. * pes[1].std + .499);
                    if (pes[1].low < 1) pes[1].low = 1;
                    if (*p != 0 && ispunct(*p) && isdigit(p[1]))
                      pes[1].high = (int)(strtod(p+1, &p) + .499);
                    if (*p != 0 && ispunct(*p) && isdigit(p[1]))
                      pes[1].low  = (int)(strtod(p+1, &p) + .499);
                    VLOG(INFO) << "mean insert size: " << pes[1].avg << ", stddev: " 
                      << pes[1].std << ", max: " << pes[1].high << ", min: " << pes[1].low;
                  }
                  else return Internal("unrecognized bwa arg ");;
                }
                if (mode != "") {
                  if (mode == "intractg")  {
                    if (!opt0.o_del) opt->o_del = 16;
                    if (!opt0.o_ins) opt->o_ins = 16;
                    if (!opt0.b) opt->b = 9;
                    if (!opt0.pen_clip5) opt->pen_clip5 = 5;
                    if (!opt0.pen_clip3) opt->pen_clip3 = 5;
                  } else if (mode == "pacbio" || mode == "pbref" || mode == "ont2d") {
                    if (!opt0.o_del) opt->o_del = 1;
                    if (!opt0.e_del) opt->e_del = 1;
                    if (!opt0.o_ins) opt->o_ins = 1;
                    if (!opt0.e_ins) opt->e_ins = 1;
                    if (!opt0.b) opt->b = 1;
                    if (opt0.split_factor == 0.) opt->split_factor = 10.;
                    if (mode == "ont2d") {
                      if (!opt0.min_chain_weight) opt->min_chain_weight = 20;
                      if (!opt0.min_seed_len) opt->min_seed_len = 14;
                      if (!opt0.pen_clip5) opt->pen_clip5 = 0;
                      if (!opt0.pen_clip3) opt->pen_clip3 = 0;
                    } else {
                      if (!opt0.min_chain_weight) opt->min_chain_weight = 40;
                      if (!opt0.min_seed_len) opt->min_seed_len = 17;
                      if (!opt0.pen_clip5) opt->pen_clip5 = 0;
                      if (!opt0.pen_clip3) opt->pen_clip3 = 0;
                    }
                  } else {
                    fprintf(stderr, "[E::%s] unknown read type '%s'\n", __func__, mode.c_str());
                    return Internal("BWA options error parsing mode"); // FIXME memory leak
                  }
                } else update_a(opt, &opt0);
                bwa_fill_scmat(opt->a, opt->b, opt->mat);


                unique_ptr<mem_opt_t> value(opt);
                *options = new BWAOptionsContainer(move(value));
                return Status::OK();
            };

            TF_RETURN_IF_ERROR(
                cinfo_.resource_manager()->LookupOrCreate<BWAOptionsContainer>(
                    cinfo_.container(), cinfo_.name(), &bwa_options, creator));

            auto h = options_handle_.AccessTensor(ctx)->flat<string>();
            h(0) = cinfo_.container();
            h(1) = cinfo_.name();
            options_handle_set_ = true;
            return Status::OK();
        }

        mutex mu_;
        std::vector<string> options_;
        PersistentTensor options_handle_ GUARDED_BY(mu_);
        bool options_handle_set_ GUARDED_BY(mu_);
    };

    REGISTER_OP("BWAOptions")
        .Output("handle: Ref(string)")
        .Attr("options: list(string)")
        .Attr("container: string = ''")
        .Attr("shared_name: string = ''")
        .SetIsStateful()
        .Doc(R"doc(
    An op that creates or gives ref to a bwa index.
    handle: The handle to the BWAOptions resource.
    genome_location: The path to the genome index directory.
    container: If non-empty, this index is placed in the given container.
    Otherwise, a default container is used.
    shared_name: If non-empty, this queue will be shared under the given name
    across multiple sessions.
    )doc");

    REGISTER_KERNEL_BUILDER(Name("BWAOptions").Device(DEVICE_CPU), BWAOptionsOp);
}  // namespace tensorflow
