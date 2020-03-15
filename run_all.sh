for body_part in XR_WRIST XR_ELBOW XR_FINGER XR_FOREARM XR_HAND XR_HUMERUS XR_SHOULDER; do
    echo "Training for $body_part"
    python main.py $body_part
done