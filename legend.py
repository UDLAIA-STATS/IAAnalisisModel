    # Pr Probabilidad de que haya un objeto de interes en la imagen
    # Bx, By -> Centro en x y y del bounding box
    # Bw, Bh -> Ancho y alto del bounding box
    # c1, c2 -> Codificacion del tipo de objeto (etiqueta)
    # IoU -> Interseccion entre dos bounding boxes, el ideal y el predicho, mientras mas cercano a 1, mejor
    # Supresion de no maximos -> escoge el bounding box con mayor probabilidad de que haya un objeto en la imagen. Elimina los bbox con 
    # una probabilidad menos a 0.6, y del restante calcula el IoU con los bbox maximo con los restantes. Descarta los IoU
    # en un umbral mayor a 0.5. Se repiten los 2 anteriores pasos hasta que no queden mas bbox.
    # En caso de que existan mas de una categoria se realiza el proceso por categoria.
    # Si dos imagenes se superponen se debe combinar sus bbox y separarlos en dos bbox distintos. Se comparan los componentes de la 
    # bbox y se comprueba que sean diferentes.