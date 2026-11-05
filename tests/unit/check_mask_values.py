

for split in Train Val; do
    for area in Rural Urban; do
        echo "===== $split / $area ====="
        echo "Images:"
        ls "data/$split/$area/images_png" | head -5
        echo "Masks:"
        ls "data/$split/$area/masks_png" | head -5
        echo "Image count: $(find "data/$split/$area/images_png" -type f | wc -l)"
        echo "Mask count: $(find "data/$split/$area/masks_png" -type f | wc -l)"
    done
done