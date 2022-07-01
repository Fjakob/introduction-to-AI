import csp


def create_map_csp():
    domain = ['c', 'm', 'y', 'k']
    schleswig_holstein = csp.Variable("Schleswig-Holstein", domain)
    hamburg = csp.Variable("Hamburg", domain)
    mecklenburg_vorpommern = csp.Variable("Mecklenburg-Vorpommern", domain)
    niedersachsen = csp.Variable("Niedersachsen", domain)
    brandenburg = csp.Variable("Brandenburg", domain)
    bremen = csp.Variable("Bremen", domain)
    sachsen_anhalt = csp.Variable("Sachsen-Anhalt", domain)
    berlin = csp.Variable("Berlin", domain)
    nordrhein_westfalen = csp.Variable("Nordrhein-Westfalen", domain)
    hessen = csp.Variable("Hessen", domain)
    thueringen = csp.Variable("Thueringen", domain)
    sachsen = csp.Variable("Sachsen", domain)
    rheinland_pfalz = csp.Variable("Rheinland-Pfalz", domain)
    saarland = csp.Variable("Saarland", domain)
    baden_wuerttemberg = csp.Variable("Baden-Wuerttemberg", domain)
    bayern = csp.Variable("Bayern", domain)

    variables = [schleswig_holstein, hamburg, mecklenburg_vorpommern,
                 niedersachsen, brandenburg, bremen, sachsen_anhalt,
                 berlin, nordrhein_westfalen, hessen, thueringen, sachsen,
                 rheinland_pfalz, saarland, baden_wuerttemberg, bayern]

    constraints = [
        csp.UnequalConstraint(schleswig_holstein, niedersachsen),
        csp.UnequalConstraint(schleswig_holstein, hamburg),
        csp.UnequalConstraint(schleswig_holstein, mecklenburg_vorpommern),
        csp.UnequalConstraint(hamburg, niedersachsen),
        csp.UnequalConstraint(bremen, niedersachsen),
        csp.UnequalConstraint(niedersachsen, nordrhein_westfalen),
        csp.UnequalConstraint(niedersachsen, hessen),
        csp.UnequalConstraint(niedersachsen, thueringen),
        csp.UnequalConstraint(niedersachsen, sachsen_anhalt),
        csp.UnequalConstraint(niedersachsen, brandenburg),
        csp.UnequalConstraint(niedersachsen, mecklenburg_vorpommern),
        csp.UnequalConstraint(mecklenburg_vorpommern, brandenburg),
        csp.UnequalConstraint(brandenburg, berlin),
        csp.UnequalConstraint(brandenburg, sachsen_anhalt),
        csp.UnequalConstraint(brandenburg, sachsen),
        csp.UnequalConstraint(sachsen_anhalt, thueringen),
        csp.UnequalConstraint(sachsen_anhalt, sachsen),
        csp.UnequalConstraint(nordrhein_westfalen, hessen),
        csp.UnequalConstraint(nordrhein_westfalen, rheinland_pfalz),
        csp.UnequalConstraint(hessen, thueringen),
        csp.UnequalConstraint(hessen, rheinland_pfalz),
        csp.UnequalConstraint(hessen, baden_wuerttemberg),
        csp.UnequalConstraint(hessen, bayern),
        csp.UnequalConstraint(thueringen, bayern),
        csp.UnequalConstraint(thueringen, sachsen),
        csp.UnequalConstraint(sachsen, bayern),
        csp.UnequalConstraint(rheinland_pfalz, saarland),
        csp.UnequalConstraint(rheinland_pfalz, baden_wuerttemberg),
        csp.UnequalConstraint(bayern, baden_wuerttemberg)]

    return csp.ConstrainedSatisfactionProblem(variables, constraints)