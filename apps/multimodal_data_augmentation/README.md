# Multimodal Data Augmentation

Product information usually can be represented in JSON format. However, this information alone may not be sufficient to describe the characteristics of the product itself, making it difficult for users to search for and verify the product information they want.

Additionally, since LLMs generally have more extensive training data in natural language text than in JSON format, it may be challenging to efficiently extract necessary information from JSON structures.

To address this, we perform the following data processing: Based on JSON field values and product images, we create new descriptive phrases that emphasize the main features of the product. This allows us to provide users with clearer and more useful information.
