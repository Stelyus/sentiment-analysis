import model  # Your model.py file.
#
#if __name__ == '__main__':
    #arguments = docopt(__doc__)
    #print(arguments)
    #model = {}
    ## Assign model variables to commandline arguments
    #model.TRAIN_PATHS = arguments['<train_data_paths>']
    #model.BATCH_SIZE = int(arguments['--batch_size'])
    #model.HIDDEN_UNITS = [int(h) for h in arguments['--hidden_units'].split(',')]
    #model.OUTPUT_DIR = arguments['<outdir>']
    ## Run the training job
    #model.train_and_evaluate()

model.train_model()
