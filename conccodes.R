
## calculate linguistic concreteness##


##### step 1: perform on Linux terminal

## remove all lines containing xml tags
grep -vwE "(id=|no=|/s|/<normalised>|orig=|/<page>|/<text>)" 1800 > 1800s.txt

###  read the first column
awk '{ print $1, $NF }' 1800s.txt >1800s2.txt
### remove all punctuation marks:
cat 1800s2.txt | tr -d '[:punct:]' > 1800s1.txt
### remove those strings containing digits, special symbols
tr -cd "[:print:]\n" < 1800s1.txt >1800p.txt
### remove all blank lines:
sed '/^$/d' 1800p.txt > 1800p1.txt
### lowercase all words
tr '[:upper:]' '[:lower:]' < 1800p1.txt >w1800.txt
## convert all words (per line) into one string
echo $(cat w1800.txt) >s1800.txt


##### step 2: perform in fasttext, still on terminal
## run the code to obtain the 100-dim word vector
./fasttext skipgram -input <filepath>s1800.txt -output <filepath>outputfilename (e.g. f1800.vec) 



#### step 3: perform linguistic concreteness in R

### get wordVectors package from github if needed 
if(!require("wordVectors")) {
  
  install.packages("devtools")
  devtools::install_github("bmschmidt/wordVectors")
  
}

library(Matrix) ### for sparse matrices
library(wordVectors) ### cosine similarity
library(data.table) ### quickly load embeddings with fread()
library(parallel) ### faster graph construction

### functions ###

### finds nearest neighbours
get_nn <- function(i, nn, embeddings) {
  
  temp  <- acos(-pmax(pmin(cosineSimilarity(t(embeddings[i,]), embeddings), 1), 0))/pi
  index <- sort.list(temp,decreasing = T)[2:(nn+1)]
  
  cbind(i, index, temp[index])
  
}

### running the random walk ###
# seeds = vector of seed words
# transition_matrix = the output of build_graph()
# beta = beta parameter, "damping factor"
# tol = tolerance for update of ranking vector, usually only requires less than 10 iterations 

random_walk <- function(seeds, transition_matrix, beta = 0.5, tol = 1e-6) {
  
  voc <- colnames(transition_matrix)
  
  if(!all(seeds %in% colnames(transition_matrix))) {stop(paste("Seeds", seeds[!seeds %in% colnames(transition_matrix)], "are not in graph"))}
  
  P <- rep(1, length(voc))/length(voc)
  S <- ifelse(voc %in% seeds, 1/length(seeds[seeds %in% voc]), 0)*(1 - beta)
  
  repeat {
    
    P.new <- as.vector(beta*transition_matrix%*%P + S)
    
    if(sum(abs(P.new - P)) < tol) {P <- P.new;break}
    
    P <- P.new
    
  }
  
  return(P)
  
}

### graph construction, can run in parallel for speed (but this requres duplicating the embeddings in RAM for each core used)
### em = embeddings, rows correspond to words, columns to embeddings vectors
### vocab = a vector of words of length equal to the number words with embeddings, usually rownames(embeddings)
### nn = number of nearest neighbours to include in graph
### cores = if greater than 1, builds the graph in parallel for speed
build_graph <- function(em, vocab, nn = 25, cores = 6) {
  
  if(!nrow(em)==length(vocab)) {stop("number of embeddings and number of words in vocabulary do not match")}
  
  if(cores > 1) {
    
    cl <- makeCluster(getOption("cl.cores", cores))
    
    clusterExport(cl, c('em', 'get_nn', 'cosineSimilarity'))
    
    G <- parLapply(cl, 1:length(vocab), fun = get_nn, nn = 25, embeddings=em)
    
    stopCluster(cl)
    
  } else {
    
    G <- lapply(1:length(vocab), FUN = get_nn, nn = 25, embeddings=em)
    
  }
  
  E <- do.call(rbind, G)
  E <- sparseMatrix(i = E[,1], j = E[,2], x = E[,3], dims = c(length(vocab), length(vocab)), dimnames = list(vocab, vocab))
  
  D <- sparseMatrix(i=1:ncol(E),j=1:ncol(E),x=1/sqrt(rowSums(E)))
  
  Tr <- D%*%E%*%D
  
  dimnames(Tr) <- dimnames(E)
  
  return(Tr)
  
}

### my quick function for bootstrapping CIs
### pos_seeds = positive seed words
### neg_seeds = negative seed words
### trans_matrix = a transition matrix from build_graph()
### boot_seeds = number of random seed words selected in each bootstrap iteration
### boot_n = number of bootstrap iterations
### ... = other arguments to random_walk() beta and tol

bootstrap_fnc <- function(pos_seeds, neg_seeds, trans_matrix, boot_seeds, boot_n, ...) {
  
  pos <- list()
  neg <- list()
  
  for(i in 1:boot_n) {
    
    pos[[i]] <- random_walk(seeds = sample(pos_seeds, boot_seeds), transition_matrix = trans_matrix, ...)
    neg[[i]] <- random_walk(seeds = sample(neg_seeds, boot_seeds), transition_matrix = trans_matrix, ...)
    
  }
  
  final <- list()
  
  for(i in 1:boot_n) {
    
    final[[i]] <- pos[[i]]/(pos[[i]]+neg[[i]])
    
  }
  
  dat <- do.call(cbind, final)
  
  output <- data.frame(Word=colnames(Tr), Score=apply(dat, 1, mean), Score.SD=apply(dat, 1, sd),stringsAsFactors = F)
  
  return(output)
  
}



## import the data of word vector

em <- fread('f1800.vec'(fasttext output files), header = F, data.table = T, skip = 1, nrows = 3000, quote = "", sep = " ")

### need matrix for cosine distance calculation
vocab <- em$V1

em$V1 <- NULL

em <- as.matrix(em)


### make the graph and transition matrix on 2 cores
Tr <- build_graph(em = em, vocab = vocab, cores = 2, nn = 25)


#### get some seed words
conc <- fread('http://www.humanities.mcmaster.ca/~vickup/Concreteness_ratings_Brysbaert_et_al_BRM.csv')
conc <- conc[Word %in% vocab]

### take 20 most concrete and abstract as seeds
concrete <- head(conc[order(conc$Conc.M, decreasing = T),], 20)
abstract <- tail(conc[order(conc$Conc.M, decreasing = T),], 20)

test <- bootstrap_fnc(pos_seeds = concrete$Word, neg_seeds = abstract$Word, trans_matrix = Tr, boot_seeds = 15, boot_n = 50, beta = 0.99, tol = 1e-6)

test <- test[order(test$Score,decreasing=T),]

head(test, 100)
tail(test, 100)

compare <- merge(test, conc, by.x = "Word", by.y = "Word")

compare=compare[!compare$Word %in% c(concrete$Word, abstract$Word),]

mean(compare$Score)
min(compare$Score)
max(compare$Score)
mean(compare$Score.SD)



#### step 4: plot in R

pdf(file="conc.pdf", 7, 5)

xcoords=c(1660, 1670, 1680, 1690, 1700, 1710, 1720, 1730, 1740, 1750, 1760, 1770, 1780, 1790, 1800, 1810, 1820, 1830, 1840, 1850, 1860)

plot(x=c(min(xcoords), max(xcoords)), y=c(min(conc$Min_score), max(conc$Max_score)), type="n", xlab="decades (1660-1860)", ylab="Concreteness Degree", cex.lab=1)
rangecolor <- rgb(30,144,255,alpha=80,maxColorValue=255)
polygon(x=c(xcoords,rev(xcoords)),y=c(conc$Max_score,rev(conc$Mean_score)),col=rangecolor,border=NA)
polygon(x=c(xcoords,rev(xcoords)),y=c(conc$Min_score,rev(conc$Mean_score)),col=rangecolor,border=NA)
meancolor <- "blue"
lines(x=xcoords,y=conc$Mean_score,col=meancolor)
lines(x=xcoords,y=conc$Mean_score,col=meancolor, lwd=2)
legend("topright", legend="R2 is 0.02, p-value is 0.25, slope is 0.001", lty=1:1, cex=0.8)
lmout=lm(conc$Mean_score~conc$decades)
abline(lmout, col="red", lwd=3, lty=2)

dev.off ()


