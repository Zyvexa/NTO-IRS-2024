def vector_product_2d(v1, v2):
    """
    Вычисляет векторное произведение двух векторов в 2D.
    v1 и v2 - списки координат векторов.
    """
    return v1[0]*v2[1] - v1[1]*v2[0]

def point_side_2d(point1, point2, point_to_check):
    """
    Определяет, с какой стороны от прямой, образованной точками point1 и point2, лежит точка point_to_check.
    point1, point2, point_to_check - списки координат точек.
    """
    # Определяем векторы
    vector1 = [point2[0] - point1[0], point2[1] - point1[1]]
    vector2 = [point_to_check[0] - point1[0], point_to_check[1] - point1[1]]
    
    # Вычисляем векторное произведение
    cross_product = vector_product_2d(vector1, vector2)
    
    # Проверяем знак вектора
    if cross_product > 0:
        return "4"
    elif cross_product < 0:
        return "3"
    else:
        return "0"

# (442, 361) (439, 373) (130, 231)
# (421, 390) (421, 378) (130, 231)
# Пример использования
point1 = (1, 1)
point2 = (1, 2)
point_to_check = [0, 1]

print(point_side_2d(point1, point2, point_to_check))
