from passage import Cave

cave = Cave(
    filename='example.cav', #имя файла
    startname='32', #название начального пикета
    n_ring_pow=7, #колличество точек на эллипсоиде 4 * n_ring_pow
    m_per_seg=0.5 #максимальное расстояние между центрами эллипсов
)

cave.create_faces() #создание 

#cave.plot_raw_points() #визуализация исходных точек
#cave.plot_picket_rings() #визуализация точек на эллипсоидах пикетов
#cave.plot_all_rings() #визуализация точек на всех эллипсоидах

cave.create_stl('example.stl') #сохранение модели в файл
