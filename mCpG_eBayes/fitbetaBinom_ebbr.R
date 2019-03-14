library("VGAM")
library("argparse")
library("tidyverse")
library("ebbr")
library("argparser")

############################################################################
## GLOBALS
chroms_allowed <- c("chr1" , "chr2" , "chr3" , "chr4")
############################################################################
## FUNCTIONS 
loadReplicates <- function(fname1 , fname2 ) {
  #' Load two CpG report replictes into single DF
  #' No header and columns correspond to
  #' "seq" , "idx", "strand", "nMe" , "nUMe" ,"diNuc" , "triNuc"
  #' @param fname1 -character
  #' @param fname2 - character
  #' @return tm a data.frame with columns  "seq" , "idx", "strand", "nMe (summed)" , "nUMe (summed)", "total (summed)" 
  t1 <- read.csv( fname1, sep = "\t" , header= F,
                 col.names = c("seq" , "idx", "strand", "nMe_1" , "nUMe_1" ,
                               "diNuc" , "triNuc"),
                 colClasses = c("character" , "integer" , "character" , "integer" , "integer",
                                "character" , "character"))
  t2 <- read.csv( fname2, sep = "\t" , header= F,
                 col.names = c("seq" , "idx", "strand", "nMe_2" , "nUMe_2" ,
                               "diNuc" , "triNuc"),
                 colClasses = c("character" , "integer" , "character" , "integer" , "integer",
                                "character" , "character"))
  tm <- merge( t1, t2  )
  tm["nMe"] <-  tm["nMe_1"] +  tm["nMe_2"]
  tm["nUMe"] <-  tm["nUMe_1"] +  tm["nUMe_2"]
  tm <- tm[c("seq" , "idx" , "strand" , "nMe" , "nUMe")]
  tm["total"] <- tm["nMe"]  + tm["nUMe"]
  tm <- tm[ tm["total"] >0 ,  ]
  return(tm)
}

####################################################################################
## PARSE CMD Line
p <- arg_parser(description = "My description")
p <- add_argument(p ,"--f1" , help="CpG report for replicate 1. NO HEADER" )
p <- add_argument(p ,"--f2" , help="CpG report for replicate 2. NO HEADER" )
p <- add_argument(p ,"--fo" , help="basename for output")
argv <- parse_args( p)
            
#argv <- parse_args( p, c("--f1" ,"./CpGRpt_3A23L_s1.sorted.txt",
#                           "--f2" , "./CpGRpt_3A23L_s2.sorted.txt",
#                        "--fo" , "3A2L_out" )
#            )

counts_table <- loadReplicates(argv$f1 , argv$f2 )
counts_table <- subset(counts_table , seq  %in% chroms_allowed)

counts_tbl <- tibble::as_tibble(counts_table )
cat("Fitting prior\n")
ebb_pri <- ebb_fit_prior(counts_tbl, nMe,  total , method = "mle" )
cat(paste0( "alpha ", ebb_pri$parameters$alpha,
            "\nbeta " ,  ebb_pri$parameters$beta ),
      file = paste0(argv$fo , ".prior.txt"))
cat("getting posterior\n")
post_out <- add_ebb_estimate(counts_tbl, nMe,  total , method = "mle")

write_tsv( post_out, paste0(argv$fo , ".posterior.tsv") )

