from CatPredictor import CatPredictor

# define example image
filename = 'images/ea36b20b28fd093ed1584d05fb1d4e9fe777ead218ac104497f5c978a7eebdbb_640.jpg'

# instantiate CatPredictor object
cat_pred = CatPredictor()
# use object to predict labels for image
results = cat_pred.predict(filename)

# print out the results for both models
print('\nVGG16:')
cat_pred.print_predictions(results['vgg16'])

print('\nMobileNet:')
cat_pred.print_predictions(results['mobilenet'])
