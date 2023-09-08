# Structure
- mycodefull.py includes the code of the authors with some minor adaptations
- folder "splitted" includes the same code, but splitted in several files and with renamed variables (they match with the authors variable names in papers' appendix) -> cleaner structure
  - the main file is run.py; you need to run this file to run the whole model

# Requirements
I recommend using conda, not pip as package manager
1. Install conda
2. Install JAVA and set PATH (google for this)
3. Create conda environment and install PySpark (see https://spark.apache.org/docs/latest/api/python/getting_started/install.html#using-conda)
4. Install R
5. Install R package forecast via terminal:
   6. Start R-Mode in terminal: R
   7. Install forecast package: install.packages('forecast', dependencies=TRUE)
   8. Install polynom package: install.packages('polynom', dependencies=TRUE)
9. Adapt file paths in splitted/settings.py (replace paths which includes my name "ole")
10. Now you should be able to run the run.py file in splitted/run.py