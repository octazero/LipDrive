import UIKit

//This is related to Profile Details Cell XIB file
class ProfileTableViewCell: UITableViewCell {
    @IBOutlet weak var ProfilePictureImageView: UIImageView!
    @IBOutlet weak var NameLabel: UILabel!
    @IBOutlet weak var EmailLabel: UILabel!
    
    
    override func awakeFromNib() {
        super.awakeFromNib()
        
        //Initialize the profile picture radius and border
        self.ProfilePictureImageView.layer.cornerRadius = self.ProfilePictureImageView.frame.width/2
        self.ProfilePictureImageView.clipsToBounds = true
        self.ProfilePictureImageView.layer.borderColor = UIColor.white.cgColor
        self.ProfilePictureImageView.layer.borderWidth = 1.0
    }

    override func setSelected(_ selected: Bool, animated: Bool) {
        super.setSelected(selected, animated: animated)

        // Configure the view for the selected state
    }
    
}
