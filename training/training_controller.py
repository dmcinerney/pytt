# trains model in sections, copying the checkpoint folder for each
# loads raw dataset(s)
# creates batcher using custom function
# for each section
#   creates/loads model, optimizer, batch_iterator
#   calls train
#   copy checkpoint directory