ABC_Data <- read_excel("D:/DAta/R/ABC_Data.xlsx")

ABC_Data$Business_Unit<-as.numeric(as.factor(ABC_Data$Business_Unit))
ABC_Data$Product_Held <- as.numeric(as.factor(ABC_Data$Product_Held))
View(ABC_Data)
ABCData.fa <- factanal(ABC_Data, factors = 2)
my_pca <- prcomp(ABC_Data, scale = TRUE,
                 center = TRUE, retx = T)
names(my_pca)
# View the principal component loading
my_pca$rotation
# See the principal components
dim(my_pca$x)
my_pca$x
# Plotting the resultant principal components
# The parameter scale = 0 ensures that arrows
# are scaled to represent the loadings
biplot(my_pca, main = "Biplot", scale = 0)
# Compute standard deviation
my_pca$sdev
# Compute variance
my_pca.var <- my_pca$sdev^2
my_pca.var
# Proportion of variance for a scree plot
propve <- my_pca.var / sum(my_pca.var)
propve
# Plot variance explained for each principal component
plot(propve, xlab = "principal component",
     ylab = "Proportion of Variance Explained",
     ylim = c(0, 1), type = "b",
     main = "Scree Plot")
# Plot the cumulative proportion of variance explained
plot(cumsum(propve),
     xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     ylim = c(0, 1), type = "b")
# Find Top n principal component
# which will atleast cover 90 % variance of dimension
which(cumsum(propve) >= 0.9)[1]

